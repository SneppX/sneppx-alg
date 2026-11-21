#![allow(non_camel_case_types)]

use std::collections::HashMap;
use std::ffi::CStr;
use std::net::SocketAddr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time;

// ============================================================================
// Public C-compatible types
// ============================================================================

#[repr(C)]
pub enum ReduceOp {
    SUM = 0,
    AVG = 1,
    MAX = 2,
    MIN = 3,
}

#[repr(C)]
pub enum DistError {
    SUCCESS = 0,
    ERR_NOT_INIT = -1,
    ERR_ALREADY_INIT = -2,
    ERR_CONNECT = -3,
    ERR_DISCOVERY = -4,
    ERR_REDUCE = -5,
    ERR_PARAM_SYNC = -6,
    ERR_NO_PEERS = -7,
    ERR_NULL_PTR = -8,
    ERR_RING_TOPO = -9,
    ERR_TIMEOUT = -10,
    ERR_INTERNAL = -99,
}

#[repr(C)]
pub struct DistConfig {
    pub port: u16,
    pub heartbeat_interval_ms: u64,
    pub heartbeat_timeout_ms: u64,
    pub max_peers: u32,
    pub enable_compression: u8,
    pub top_k_ratio: f32,
    pub enable_privacy: u8,
    pub privacy_epsilon: f64,
    pub privacy_delta: f64,
    pub gradient_clip_norm: f32,
}

impl Default for DistConfig {
    fn default() -> Self {
        Self {
            port: 9876,
            heartbeat_interval_ms: 1000,
            heartbeat_timeout_ms: 5000,
            max_peers: 32,
            enable_compression: 1,
            top_k_ratio: 0.01,
            enable_privacy: 0,
            privacy_epsilon: 1.0,
            privacy_delta: 1e-5,
            gradient_clip_norm: 1.0,
        }
    }
}

// ============================================================================
// Internal types
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ProtoMsg {
    DiscoveryReq { node_id: u64, listen_port: u16 },
    DiscoveryResp { peers: Vec<PeerAddr> },
    GradientChunk { chunk_id: u32, total_chunks: u32, data: Vec<f32>, compressed: bool, indices: Vec<u32> },
    ParamSync { name: String, data: Vec<f32>, version: u64, node_id: u64 },
    Heartbeat { node_id: u64, timestamp: u64 },
    HeartbeatAck { node_id: u64 },
    PeerLeave { node_id: u64 },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct PeerAddr {
    node_id: u64,
    addr: String,
    port: u16,
}

struct PeerState {
    node_id: u64,
    addr: SocketAddr,
    stream: tokio::sync::Mutex<Option<TcpStream>>,
    last_heartbeat: Mutex<Instant>,
    failed: AtomicBool,
    version: AtomicU64,
    reconnect_backoff: Mutex<Duration>,
}

#[derive(Clone)]
struct RingTopology {
    peers: Vec<SocketAddr>,
    node_ids: Vec<u64>,
    rank: usize,
    world_size: usize,
}

#[derive(Debug, Clone)]
struct ParamState {
    data: Vec<f32>,
    version: u64,
    node_id: u64,
    updated_at: Instant,
}

struct ParamStore {
    params: HashMap<String, ParamState>,
    local_version: AtomicU64,
}

// ============================================================================
// Global runtime state
// ============================================================================

struct DistRuntime {
    rt: tokio::runtime::Runtime,
    state: Arc<SharedState>,
}

struct SharedState {
    config: DistConfig,
    node_id: AtomicU64,
    port: u16,
    running: AtomicBool,
    peers: RwLock<Vec<Arc<PeerState>>>,
    topology: RwLock<Option<RingTopology>>,
    param_store: RwLock<ParamStore>,
    discovery_tx: Mutex<Option<mpsc::Sender<PeerAddr>>>,
    heartbeat_tx: Mutex<Option<mpsc::Sender<u64>>>,
}

static GLOBAL: Mutex<Option<DistRuntime>> = Mutex::new(None);

fn next_node_id() -> u64 {
    let mut rng = rand::thread_rng();
    rng.gen::<u64>()
}

// ============================================================================
// Helpers
// ============================================================================

fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn with_runtime<F, R>(f: F) -> Result<R, DistError>
where
    F: FnOnce(&DistRuntime) -> Result<R, DistError>,
{
    let guard = GLOBAL.lock().unwrap();
    guard.as_ref().map(f).ok_or(DistError::ERR_NOT_INIT)?
}

fn read_f32_slice(ptr: *mut f32, count: usize) -> Result<Vec<f32>, DistError> {
    if ptr.is_null() {
        return Err(DistError::ERR_NULL_PTR);
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec())
}

fn write_f32_slice(ptr: *mut f32, data: &[f32]) -> Result<(), DistError> {
    if ptr.is_null() {
        return Err(DistError::ERR_NULL_PTR);
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) };
    dst.copy_from_slice(data);
    Ok(())
}

async fn send_msg(stream: &mut TcpStream, msg: &ProtoMsg) -> Result<(), DistError> {
    let payload = bincode::serialize(msg).map_err(|_| DistError::ERR_INTERNAL)?;
    let len = (payload.len() as u32).to_le_bytes();
    stream.write_all(&len).await.map_err(|_| DistError::ERR_CONNECT)?;
    stream.write_all(&payload).await.map_err(|_| DistError::ERR_CONNECT)?;
    Ok(())
}

async fn recv_msg(stream: &mut TcpStream) -> Result<ProtoMsg, DistError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.map_err(|_| DistError::ERR_CONNECT)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await.map_err(|_| DistError::ERR_CONNECT)?;
    bincode::deserialize(&payload).map_err(|_| DistError::ERR_INTERNAL)
}

async fn get_peer_stream_by_addr(
    peers: &[Arc<PeerState>],
    target: SocketAddr,
) -> Result<tokio::sync::MutexGuard<'_, Option<TcpStream>>, DistError> {
    for p in peers {
        if p.addr == target {
            let guard = p.stream.lock().await;
            if guard.is_some() {
                return Ok(guard);
            }
        }
    }
    Err(DistError::ERR_NO_PEERS)
}

async fn connect_to_peer_inner(
    state: Arc<SharedState>,
    addr_str: &str,
    port: u16,
) -> Result<(), DistError> {
    let sock_addr: SocketAddr = format!("{}:{}", addr_str, port)
        .parse()
        .map_err(|_| DistError::ERR_CONNECT)?;

    let mut stream = TcpStream::connect(sock_addr)
        .await
        .map_err(|_| DistError::ERR_CONNECT)?;

    let node_id = state.node_id.load(Ordering::Relaxed);
    let disc_msg = ProtoMsg::DiscoveryReq {
        node_id,
        listen_port: state.port,
    };
    send_msg(&mut stream, &disc_msg).await?;

    let resp = recv_msg(&mut stream).await?;
    if let ProtoMsg::DiscoveryResp { peers } = resp {
        let mut new_peers = Vec::new();
        {
            let mut peer_guard = state.peers.write().unwrap();
            for p in &peers {
                let exists = peer_guard.iter().any(|x| x.node_id == p.node_id);
                if !exists && peer_guard.len() < state.config.max_peers as usize {
                    let sock: SocketAddr = format!("{}:{}", p.addr, p.port).parse().unwrap();
                    let ps = Arc::new(PeerState {
                        node_id: p.node_id,
                        addr: sock,
                        stream: tokio::sync::Mutex::new(None),
                        last_heartbeat: Mutex::new(Instant::now()),
                        failed: AtomicBool::new(false),
                        version: AtomicU64::new(0),
                        reconnect_backoff: Mutex::new(Duration::from_millis(100)),
                    });
                    new_peers.push((p.node_id, ps.clone()));
                    peer_guard.push(ps);
                }
            }
        }
        for (_nid, ps) in &new_peers {
            let sock = ps.addr;
            let ps_c = ps.clone();
            tokio::spawn(async move {
                match TcpStream::connect(sock).await {
                    Ok(s) => {
                        *ps_c.stream.lock().await = Some(s);
                    }
                    Err(_) => {
                        ps_c.failed.store(true, Ordering::Relaxed);
                    }
                }
            });
        }
    }

    {
        let mut peer_guard = state.peers.write().unwrap();
        let exists = peer_guard.iter().any(|p| p.node_id == node_id);
        if !exists && peer_guard.len() < state.config.max_peers as usize {
            let ps = Arc::new(PeerState {
                node_id,
                addr: sock_addr,
                stream: tokio::sync::Mutex::new(Some(stream)),
                last_heartbeat: Mutex::new(Instant::now()),
                failed: AtomicBool::new(false),
                version: AtomicU64::new(0),
                reconnect_backoff: Mutex::new(Duration::from_millis(100)),
            });
            peer_guard.push(ps);
        }
    }

    Ok(())
}

// ============================================================================
// Node discovery
// ============================================================================

async fn discovery_server(state: Arc<SharedState>) {
    let port = state.port;
    let listener = match TcpListener::bind(format!("0.0.0.0:{}", port)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[arx_dist] discovery bind failed on port {}: {}", port, e);
            return;
        }
    };

    while state.running.load(Ordering::Relaxed) {
        let (mut stream, addr) = match listener.accept().await {
            Ok(conn) => conn,
            Err(_) => continue,
        };

        let state_clone = state.clone();
        tokio::spawn(async move {
            if let Ok(msg) = recv_msg(&mut stream).await {
                match msg {
                    ProtoMsg::DiscoveryReq { node_id, listen_port } => {
                        {
                            let mut peers = state_clone.peers.write().unwrap();
                            let exists = peers.iter().any(|p| p.node_id == node_id);
                            if !exists && peers.len() < state_clone.config.max_peers as usize {
                                let sock: SocketAddr =
                                    format!("{}:{}", addr.ip(), listen_port).parse().unwrap();
                                let ps = Arc::new(PeerState {
                                    node_id,
                                    addr: sock,
                                    stream: tokio::sync::Mutex::new(Some(stream)),
                                    last_heartbeat: Mutex::new(Instant::now()),
                                    failed: AtomicBool::new(false),
                                    version: AtomicU64::new(0),
                                    reconnect_backoff: Mutex::new(Duration::from_millis(100)),
                                });
                                peers.push(ps);
                            }
                        }

                        let peer_list: Vec<PeerAddr> = {
                            let peers = state_clone.peers.read().unwrap();
                            peers
                                .iter()
                                .filter(|p| p.node_id != node_id)
                                .map(|p| PeerAddr {
                                    node_id: p.node_id,
                                    addr: p.addr.ip().to_string(),
                                    port: p.addr.port(),
                                })
                                .collect()
                        };

                        let resp = ProtoMsg::DiscoveryResp { peers: peer_list };
                        if let Ok(mut stream) = TcpStream::connect(addr).await {
                            let _ = send_msg(&mut stream, &resp).await;
                        }
                    }
                    _ => {}
                }
            }
        });
    }
}

// ============================================================================
// Heartbeat / fault tolerance
// ============================================================================

async fn heartbeat_loop(state: Arc<SharedState>) {
    let interval_ms = state.config.heartbeat_interval_ms.max(100);
    let timeout_ms = state.config.heartbeat_timeout_ms.max(500);
    let mut ticker = time::interval(Duration::from_millis(interval_ms));
    let node_id = state.node_id.load(Ordering::Relaxed);

    loop {
        ticker.tick().await;
        if !state.running.load(Ordering::Relaxed) {
            break;
        }

        let peers: Vec<Arc<PeerState>> = {
            let guard = state.peers.read().unwrap();
            guard.iter().filter(|p| !p.failed.load(Ordering::Relaxed)).cloned().collect()
        };

        for peer in &peers {
            let mut stream_opt = peer.stream.lock().await;
            if let Some(stream) = stream_opt.as_mut() {
                let msg = ProtoMsg::Heartbeat { node_id, timestamp: now_micros() };
                if send_msg(stream, &msg).await.is_err() {
                    *stream_opt = None;
                    peer.failed.store(true, Ordering::Relaxed);
                }
            } else {
                peer.failed.store(true, Ordering::Relaxed);
            }
        }

        // Check for timeouts
        let now = Instant::now();
        {
            let guard = state.peers.read().unwrap();
            for peer in guard.iter() {
                if !peer.failed.load(Ordering::Relaxed) {
                    let last = *peer.last_heartbeat.lock().unwrap();
                    if now.duration_since(last) > Duration::from_millis(timeout_ms) {
                        peer.failed.store(true, Ordering::Relaxed);
                    }
                }
            }
        }

        // Reconnect failed peers
        let failed_peers: Vec<Arc<PeerState>> = {
            let guard = state.peers.read().unwrap();
            guard.iter().filter(|p| p.failed.load(Ordering::Relaxed)).cloned().collect()
        };

        for peer in &failed_peers {
            let peer_c = peer.clone();
            tokio::spawn(async move {
                let backoff = *peer_c.reconnect_backoff.lock().unwrap();
                time::sleep(backoff).await;
                match TcpStream::connect(peer_c.addr).await {
                    Ok(stream) => {
                        *peer_c.stream.lock().await = Some(stream);
                        peer_c.failed.store(false, Ordering::Relaxed);
                        *peer_c.last_heartbeat.lock().unwrap() = Instant::now();
                        *peer_c.reconnect_backoff.lock().unwrap() = Duration::from_millis(100);
                    }
                    Err(_) => {
                        let mut b = peer_c.reconnect_backoff.lock().unwrap();
                        *b = b.mul_f32(2.0).min(Duration::from_secs(30));
                    }
                }
            });
        }
    }
}

async fn heartbeat_listener(state: Arc<SharedState>) {
    while state.running.load(Ordering::Relaxed) {
        let peers: Vec<Arc<PeerState>> = {
            let guard = state.peers.read().unwrap();
            guard.clone()
        };

        let mut any_work = false;
        for peer in &peers {
            let mut stream_opt = peer.stream.lock().await;
            if let Some(stream) = stream_opt.as_mut() {
                match recv_msg(stream).await {
                    Ok(ProtoMsg::Heartbeat { .. }) => {
                        *peer.last_heartbeat.lock().unwrap() = Instant::now();
                        let ack =
                            ProtoMsg::HeartbeatAck { node_id: state.node_id.load(Ordering::Relaxed) };
                        let _ = send_msg(stream, &ack).await;
                        any_work = true;
                    }
                    Ok(ProtoMsg::HeartbeatAck { .. }) => {
                        *peer.last_heartbeat.lock().unwrap() = Instant::now();
                        any_work = true;
                    }
                    Ok(_) => {}
                    Err(_) => {
                        *stream_opt = None;
                        peer.failed.store(true, Ordering::Relaxed);
                    }
                }
            }
        }

        if !any_work {
            time::sleep(Duration::from_millis(50)).await;
        }
    }
}

// ============================================================================
// Gradient compression: top-k sparsification + random sampling
// ============================================================================

fn compress_topk(data: &[f32], ratio: f32) -> (Vec<u32>, Vec<f32>) {
    let n = data.len();
    if n == 0 || ratio <= 0.0 {
        return (Vec::new(), Vec::new());
    }
    let k = ((n as f32) * ratio).max(1.0).min(n as f32) as usize;

    let mut indices: Vec<u32> = (0..n as u32).collect();
    indices.sort_by(|a, b| {
        let va = data[*a as usize].abs();
        let vb = data[*b as usize].abs();
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });

    let topk_idx: Vec<u32> = indices[..k].to_vec();
    let topk_vals: Vec<f32> = topk_idx.iter().map(|i| data[*i as usize]).collect();

    let remaining: Vec<u32> = indices[k..].to_vec();
    let mut rng = rand::thread_rng();
    let sampled: Vec<u32> = remaining
        .iter()
        .filter(|_| rng.gen::<f32>() < 0.5)
        .take((remaining.len() / 2).max(1))
        .copied()
        .collect();

    let sampled_vals: Vec<f32> = sampled.iter().map(|i| {
        let scale = if !sampled.is_empty() { n as f32 / sampled.len() as f32 } else { 1.0 };
        data[*i as usize] * scale
    }).collect();

    let mut out_idx = topk_idx;
    let mut out_vals = topk_vals;
    out_idx.extend(&sampled);
    out_vals.extend(&sampled_vals);

    (out_idx, out_vals)
}

fn decompress(data: &mut [f32], indices: &[u32], values: &[f32]) {
    for (i, val) in indices.iter().zip(values.iter()) {
        if (*i as usize) < data.len() {
            data[*i as usize] = *val;
        }
    }
}

// ============================================================================
// Ring all-reduce
// ============================================================================

async fn ring_all_reduce_impl(
    state: Arc<SharedState>,
    data: &mut [f32],
) -> Result<(), DistError> {
    let topology = {
        let topo = state.topology.read().unwrap();
        topo.clone().ok_or(DistError::ERR_RING_TOPO)?
    };

    let world_size = topology.world_size;
    if world_size < 2 {
        return Ok(());
    }

    let rank = topology.rank;
    let n = data.len();
    let chunk_size = (n + world_size - 1) / world_size;
    let enable_compression = state.config.enable_compression != 0;
    let top_k_ratio = state.config.top_k_ratio;

    let pred = (rank + world_size - 1) % world_size;
    let succ = (rank + 1) % world_size;

    let pred_addr = topology.peers[pred];
    let succ_addr = topology.peers[succ];

    // Helper: send chunk to successor, receive from predecessor
    async fn scatter_reduce_step(
        peers: &[Arc<PeerState>],
        data: &mut [f32],
        rank: usize,
        world_size: usize,
        chunk_size: usize,
        n: usize,
        step: usize,
        enable_compression: bool,
        top_k_ratio: f32,
        pred_addr: SocketAddr,
        succ_addr: SocketAddr,
    ) -> Result<(), DistError> {
        let send_chunk_idx = (rank + world_size - step) % world_size;
        let recv_chunk_idx = (rank + world_size - step - 1) % world_size;

        let send_start = send_chunk_idx * chunk_size;
        let send_end = (send_start + chunk_size).min(n);
        let send_data: Vec<f32> = data[send_start..send_end].to_vec();

        let (compressed_idx, compressed_vals) = if enable_compression {
            compress_topk(&send_data, top_k_ratio)
        } else {
            (Vec::new(), send_data)
        };

        let msg = ProtoMsg::GradientChunk {
            chunk_id: send_chunk_idx as u32,
            total_chunks: world_size as u32,
            data: compressed_vals,
            compressed: enable_compression,
            indices: compressed_idx,
        };

        // Send to successor
        {
            let mut guard = get_peer_stream_by_addr(peers, succ_addr).await?;
            let stream = guard.as_mut().unwrap();
            send_msg(stream, &msg).await?;
        }

        // Receive from predecessor
        let recv_msg = {
            let mut guard = get_peer_stream_by_addr(peers, pred_addr).await?;
            let stream = guard.as_mut().unwrap();
            recv_msg(stream).await?
        };

        if let ProtoMsg::GradientChunk { chunk_id: _, data: recv_data, compressed, indices, .. } = recv_msg {
            let recv_start = (recv_chunk_idx * chunk_size).min(n);
            let recv_end = (recv_start + recv_data.len()).min(n);
            let dest = &mut data[recv_start..recv_end];

            if compressed && !indices.is_empty() {
                let mut decomp = vec![0.0f32; dest.len()];
                decompress(&mut decomp, &indices, &recv_data);
                for (j, v) in decomp.iter().enumerate() {
                    if j < dest.len() {
                        dest[j] += v;
                    }
                }
            } else {
                for (j, v) in recv_data.iter().enumerate() {
                    if j < dest.len() {
                        dest[j] += v;
                    }
                }
            }
        }

        Ok(())
    }

    async fn all_gather_step(
        peers: &[Arc<PeerState>],
        data: &mut [f32],
        rank: usize,
        world_size: usize,
        chunk_size: usize,
        n: usize,
        step: usize,
        enable_compression: bool,
        top_k_ratio: f32,
        pred_addr: SocketAddr,
        succ_addr: SocketAddr,
    ) -> Result<(), DistError> {
        let send_chunk_idx = (rank + world_size - step) % world_size;
        let recv_chunk_idx = (rank + world_size - step - 1) % world_size;

        let send_start = send_chunk_idx * chunk_size;
        let send_end = (send_start + chunk_size).min(n);
        let send_slice = &data[send_start..send_end];

        let (cidx, cvals) = if enable_compression {
            compress_topk(send_slice, top_k_ratio * 0.5)
        } else {
            (Vec::new(), send_slice.to_vec())
        };

        let msg = ProtoMsg::GradientChunk {
            chunk_id: send_chunk_idx as u32,
            total_chunks: world_size as u32,
            data: cvals,
            compressed: enable_compression,
            indices: cidx,
        };

        {
            let mut guard = get_peer_stream_by_addr(peers, succ_addr).await?;
            let stream = guard.as_mut().unwrap();
            send_msg(stream, &msg).await?;
        }

        let recv_msg = {
            let mut guard = get_peer_stream_by_addr(peers, pred_addr).await?;
            let stream = guard.as_mut().unwrap();
            recv_msg(stream).await?
        };

        if let ProtoMsg::GradientChunk { chunk_id: _, data: recv_data, compressed, indices, .. } = recv_msg {
            let recv_start = (recv_chunk_idx * chunk_size).min(n);
            let recv_end = (recv_start + recv_data.len()).min(n);
            let dest = &mut data[recv_start..recv_end];

            if compressed && !indices.is_empty() {
                let mut decomp = vec![0.0f32; dest.len()];
                decompress(&mut decomp, &indices, &recv_data);
                let copy_len = dest.len().min(decomp.len());
                dest[..copy_len].copy_from_slice(&decomp[..copy_len]);
            } else {
                for (j, v) in recv_data.iter().enumerate() {
                    if j < dest.len() {
                        dest[j] = *v;
                    }
                }
            }
        }

        Ok(())
    }

    let peers_guard = state.peers.read().unwrap();
    let peers: Vec<Arc<PeerState>> = peers_guard.clone();
    drop(peers_guard);

    // Scatter-reduce phase
    for step in 0..world_size - 1 {
        scatter_reduce_step(
            &peers, data, rank, world_size, chunk_size, n, step,
            enable_compression, top_k_ratio, pred_addr, succ_addr,
        )
        .await?;
    }

    // All-gather phase
    for step in 0..world_size - 1 {
        all_gather_step(
            &peers, data, rank, world_size, chunk_size, n, step,
            enable_compression, top_k_ratio, pred_addr, succ_addr,
        )
        .await?;
    }

    if world_size > 0 {
        let scale = 1.0 / world_size as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
    }

    Ok(())
}

async fn all_reduce_impl(
    state: Arc<SharedState>,
    data: &mut [f32],
    _op: ReduceOp,
) -> Result<(), DistError> {
    ring_all_reduce_impl(state, data).await
}

// ============================================================================
// Parameter sync
// ============================================================================

async fn sync_params_impl(
    state: Arc<SharedState>,
    param_ptrs: &[*mut f32],
    shapes: &[usize],
) -> Result<(), DistError> {
    let node_id = state.node_id.load(Ordering::Relaxed);
    let peers: Vec<Arc<PeerState>> = {
        let guard = state.peers.read().unwrap();
        guard.iter().filter(|p| !p.failed.load(Ordering::Relaxed)).cloned().collect()
    };

    if peers.is_empty() {
        return Ok(());
    }

    let param_names: Vec<String> =
        (0..param_ptrs.len()).map(|i| format!("param_{}", i)).collect();

    let param_snapshots: Vec<ProtoMsg> = param_ptrs
        .iter()
        .enumerate()
        .map(|(i, ptr)| {
            let slice = unsafe { std::slice::from_raw_parts(*ptr, shapes[i]) };
            let version = {
                let store = state.param_store.read().unwrap();
                store.params.get(&param_names[i]).map(|p| p.version).unwrap_or(0)
            };
            ProtoMsg::ParamSync {
                name: param_names[i].clone(),
                data: slice.to_vec(),
                version: version + 1,
                node_id,
            }
        })
        .collect();

    // Send to all peers
    for peer in &peers {
        let mut guard = peer.stream.lock().await;
        if let Some(stream) = guard.as_mut() {
            for msg in &param_snapshots {
                if send_msg(stream, msg).await.is_err() {
                    peer.failed.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
    }

    // Collect from all peers
    for peer in &peers {
        let mut guard = peer.stream.lock().await;
        if let Some(stream) = guard.as_mut() {
            for (i, name) in param_names.iter().enumerate() {
                match recv_msg(stream).await {
                    Ok(ProtoMsg::ParamSync { name: n, data, version, node_id: nid }) => {
                        if n != *name {
                            continue;
                        }
                        let mut store = state.param_store.write().unwrap();
                        let current = store.params.get(&n);
                        let should_update = match current {
                            None => true,
                            Some(p) => version > p.version || (version == p.version && nid > p.node_id),
                        };
                        if should_update {
                            store.params.insert(
                                n.clone(),
                                ParamState {
                                    data: data.clone(),
                                    version,
                                    node_id: nid,
                                    updated_at: Instant::now(),
                                },
                            );
                            let ptr = param_ptrs[i];
                            let len = shapes[i].min(data.len());
                            unsafe {
                                std::slice::from_raw_parts_mut(ptr, len).copy_from_slice(&data[..len]);
                            }
                        }
                    }
                    Ok(_) => continue,
                    Err(_) => {
                        peer.failed.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Differential privacy: Laplace mechanism
// ============================================================================

fn add_laplace_noise(data: &mut [f32], epsilon: f64, _delta: f64, clip_norm: f32) {
    if epsilon <= 0.0 {
        return;
    }

    let sensitivity: f64 = if clip_norm > 0.0 {
        2.0 * clip_norm as f64
    } else {
        1.0
    };

    let scale = sensitivity / epsilon;
    let mut rng = rand::thread_rng();

    for v in data.iter_mut() {
        if clip_norm > 0.0 {
            *v = v.clamp(-clip_norm, clip_norm);
        }
        let u: f64 = rng.gen::<f64>() - 0.5;
        let noise = -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln().max(-100.0);
        *v += noise as f32;
    }
}

// ============================================================================
// FFI: extern "C" API
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn arix_dist_init(config: *const DistConfig) -> i32 {
    let mut guard = GLOBAL.lock().unwrap();
    if guard.is_some() {
        return DistError::ERR_ALREADY_INIT as i32;
    }

    let cfg = if config.is_null() {
        DistConfig::default()
    } else {
        unsafe { std::ptr::read(config) }
    };

    let state = Arc::new(SharedState {
        node_id: AtomicU64::new(next_node_id()),
        port: cfg.port,
        running: AtomicBool::new(true),
        peers: RwLock::new(Vec::new()),
        topology: RwLock::new(None),
        param_store: RwLock::new(ParamStore {
            params: HashMap::new(),
            local_version: AtomicU64::new(0),
        }),
        config: cfg,
        discovery_tx: Mutex::new(None),
        heartbeat_tx: Mutex::new(None),
    });

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(_) => return DistError::ERR_INTERNAL as i32,
    };

    let disc_state = state.clone();
    rt.spawn(async move {
        discovery_server(disc_state).await;
    });

    let hb_state = state.clone();
    rt.spawn(async move {
        heartbeat_loop(hb_state).await;
    });

    let hbl_state = state.clone();
    rt.spawn(async move {
        heartbeat_listener(hbl_state).await;
    });

    *guard = Some(DistRuntime { rt, state });
    DistError::SUCCESS as i32
}

#[no_mangle]
pub extern "C" fn arix_dist_shutdown() {
    let mut guard = GLOBAL.lock().unwrap();
    if let Some(runtime) = guard.take() {
        runtime.state.running.store(false, Ordering::Relaxed);
        let peers: Vec<Arc<PeerState>> = {
            let g = runtime.state.peers.read().unwrap();
            g.clone()
        };
        for peer in &peers {
            peer.failed.store(true, Ordering::Relaxed);
        }
        drop(guard);
        runtime.rt.shutdown_timeout(Duration::from_secs(5));
    }
}

#[no_mangle]
pub unsafe extern "C" fn arix_dist_connect(addr: *const c_char, port: u16) -> i32 {
    if addr.is_null() {
        return DistError::ERR_NULL_PTR as i32;
    }
    let addr_str = match unsafe { CStr::from_ptr(addr) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return DistError::ERR_NULL_PTR as i32,
    };

    let guard = GLOBAL.lock().unwrap();
    let runtime = match guard.as_ref() {
        Some(r) => r,
        None => return DistError::ERR_NOT_INIT as i32,
    };

    let state = runtime.state.clone();
    if let Err(e) = runtime.rt.block_on(async move {
        connect_to_peer_inner(state, &addr_str, port).await
    }) {
        return e as i32;
    }

    // Build ring topology
    {
        let mut topo = runtime.state.topology.write().unwrap();
        let peers = runtime.state.peers.read().unwrap();
        let mut addrs: Vec<SocketAddr> = peers.iter().map(|p| p.addr).collect();
        let mut ids: Vec<u64> = peers.iter().map(|p| p.node_id).collect();
        let my_id = runtime.state.node_id.load(Ordering::Relaxed);
        let self_addr: SocketAddr =
            format!("127.0.0.1:{}", runtime.state.port).parse().unwrap();
        let insert_pos = addrs.iter().position(|a| *a > self_addr).unwrap_or(addrs.len());
        addrs.insert(insert_pos, self_addr);
        ids.insert(insert_pos, my_id);
        let rank = ids.iter().position(|id| *id == my_id).unwrap_or(0);
        *topo = Some(RingTopology {
            peers: addrs,
            node_ids: ids,
            rank,
            world_size: topo.as_ref().map(|t| t.world_size).unwrap_or(0) + 1,
        });
    }

    DistError::SUCCESS as i32
}

#[no_mangle]
pub unsafe extern "C" fn arix_dist_all_reduce(data: *mut f32, count: usize, op: ReduceOp) -> i32 {
    let guard = GLOBAL.lock().unwrap();
    let runtime = match guard.as_ref() {
        Some(r) => r,
        None => return DistError::ERR_NOT_INIT as i32,
    };

    let mut buf = match read_f32_slice(data, count) {
        Ok(b) => b,
        Err(e) => return e as i32,
    };

    let state = runtime.state.clone();
    let result = runtime.rt.block_on(async {
        all_reduce_impl(state, &mut buf, op).await
    });
    if let Err(e) = result {
        return e as i32;
    }

    write_f32_slice(data, &buf);
    DistError::SUCCESS as i32
}

#[no_mangle]
pub unsafe extern "C" fn arix_dist_ring_all_reduce(data: *mut f32, count: usize) -> i32 {
    let guard = GLOBAL.lock().unwrap();
    let runtime = match guard.as_ref() {
        Some(r) => r,
        None => return DistError::ERR_NOT_INIT as i32,
    };

    let mut buf = match read_f32_slice(data, count) {
        Ok(b) => b,
        Err(e) => return e as i32,
    };

    let state = runtime.state.clone();
    let result = runtime.rt.block_on(async {
        ring_all_reduce_impl(state, &mut buf).await
    });
    if let Err(e) = result {
        return e as i32;
    }

    write_f32_slice(data, &buf);
    DistError::SUCCESS as i32
}

#[no_mangle]
pub unsafe extern "C" fn arix_dist_sync_params(
    params: *mut *mut f32,
    shapes: *mut usize,
    n: usize,
) -> i32 {
    if params.is_null() || shapes.is_null() {
        return DistError::ERR_NULL_PTR as i32;
    }

    let guard = GLOBAL.lock().unwrap();
    let runtime = match guard.as_ref() {
        Some(r) => r,
        None => return DistError::ERR_NOT_INIT as i32,
    };

    let ptrs = unsafe { std::slice::from_raw_parts(params, n) };
    let szs = unsafe { std::slice::from_raw_parts(shapes, n) };

    let state = runtime.state.clone();
    if let Err(e) = runtime.rt.block_on(async move {
        sync_params_impl(state, ptrs, szs).await
    }) {
        return e as i32;
    }

    DistError::SUCCESS as i32
}

#[no_mangle]
pub unsafe extern "C" fn arix_dist_add_noise(
    data: *mut f32,
    count: usize,
    epsilon: f64,
    delta: f64,
) {
    if data.is_null() || count == 0 {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(data, count) };
    let clip_norm = GLOBAL
        .lock()
        .unwrap()
        .as_ref()
        .map(|r| r.state.config.gradient_clip_norm)
        .unwrap_or(1.0);
    add_laplace_noise(slice, epsilon, delta, clip_norm);
}

// ============================================================================
// Rust public API
// ============================================================================

pub fn init(config: DistConfig) -> Result<(), DistError> {
    let result = unsafe { arix_dist_init(&config as *const DistConfig) };
    if result >= 0 {
        Ok(())
    } else {
        Err(unsafe { std::mem::transmute::<i32, DistError>(result as i32) })
    }
}

pub fn shutdown() {
    arix_dist_shutdown()
}

pub fn connect(addr: &str, port: u16) -> Result<(), DistError> {
    let c_addr = std::ffi::CString::new(addr).map_err(|_| DistError::ERR_CONNECT)?;
    let result = unsafe { arix_dist_connect(c_addr.as_ptr(), port) };
    if result >= 0 {
        Ok(())
    } else {
        Err(unsafe { std::mem::transmute::<i32, DistError>(result) })
    }
}

pub fn all_reduce(data: &mut [f32], op: ReduceOp) -> Result<(), DistError> {
    let result = unsafe { arix_dist_all_reduce(data.as_mut_ptr(), data.len(), op) };
    if result >= 0 {
        Ok(())
    } else {
        Err(unsafe { std::mem::transmute::<i32, DistError>(result) })
    }
}

pub fn ring_all_reduce(data: &mut [f32]) -> Result<(), DistError> {
    let result = unsafe { arix_dist_ring_all_reduce(data.as_mut_ptr(), data.len()) };
    if result >= 0 {
        Ok(())
    } else {
        Err(unsafe { std::mem::transmute::<i32, DistError>(result) })
    }
}

pub fn sync_params(params: &mut [&mut [f32]]) -> Result<(), DistError> {
    let mut ptrs: Vec<*mut f32> = params.iter_mut().map(|p| p.as_mut_ptr()).collect();
    let mut shapes: Vec<usize> = params.iter().map(|p| p.len()).collect();
    let result =
        unsafe { arix_dist_sync_params(ptrs.as_mut_ptr(), shapes.as_mut_ptr(), params.len()) };
    if result >= 0 {
        Ok(())
    } else {
        Err(unsafe { std::mem::transmute::<i32, DistError>(result) })
    }
}

pub fn add_noise(data: &mut [f32], epsilon: f64, delta: f64) {
    unsafe { arix_dist_add_noise(data.as_mut_ptr(), data.len(), epsilon, delta) }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist_config_default() {
        let cfg = DistConfig::default();
        assert_eq!(cfg.port, 9876);
        assert_eq!(cfg.heartbeat_interval_ms, 1000);
        assert!(cfg.enable_compression != 0);
    }

    #[test]
    fn test_laplace_noise() {
        let mut data = vec![1.0f32; 100];
        add_laplace_noise(&mut data, 1.0, 1e-5, 1.0);
        let mean = data.iter().copied().sum::<f32>() / data.len() as f32;
        assert!((mean - 1.0).abs() < 0.5, "mean drifted: {}", mean);
    }

    #[test]
    fn test_topk_compression() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let (idx, vals) = compress_topk(&data, 0.1);
        assert!(!idx.is_empty());
        assert_eq!(idx.len(), vals.len());
        assert!(idx.len() < data.len());
        assert!(vals.contains(&99.0));
    }

    #[test]
    fn test_decompress() {
        let mut data = vec![0.0f32; 10];
        let indices = vec![1u32, 3, 5];
        let values = vec![10.0f32, 30.0, 50.0];
        decompress(&mut data, &indices, &values);
        assert_eq!(data[1], 10.0);
        assert_eq!(data[3], 30.0);
        assert_eq!(data[5], 50.0);
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(DistError::SUCCESS as i32, 0);
        assert_eq!(DistError::ERR_NOT_INIT as i32, -1);
        assert_eq!(DistError::ERR_NULL_PTR as i32, -8);
    }

    #[test]
    fn test_reduce_op_values() {
        assert_eq!(ReduceOp::SUM as i32, 0);
        assert_eq!(ReduceOp::AVG as i32, 1);
        assert_eq!(ReduceOp::MAX as i32, 2);
        assert_eq!(ReduceOp::MIN as i32, 3);
    }
}
