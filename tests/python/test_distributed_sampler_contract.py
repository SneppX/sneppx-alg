"""Contract tests shared by the distributed sampler entry points."""

import pytest

from SneppX_ALG.interface_bindings import distributed, nccl


@pytest.mark.parametrize(
    "sampler_type", [distributed.DistributedSampler, nccl.DistributedSampler]
)
def test_non_divisible_dataset_is_evenly_padded(sampler_type):
    dataset = list(range(10))
    shards = [
        list(sampler_type(dataset, num_replicas=3, rank=rank, shuffle=False))
        for rank in range(3)
    ]

    assert [len(shard) for shard in shards] == [4, 4, 4]
    assert sorted(set(value for shard in shards for value in shard)) == dataset
    assert sum(len(shard) for shard in shards) == 12


@pytest.mark.parametrize(
    "sampler_type", [distributed.DistributedSampler, nccl.DistributedSampler]
)
def test_drop_last_produces_equal_complete_shards(sampler_type):
    dataset = list(range(10))
    shards = [
        list(
            sampler_type(
                dataset, num_replicas=3, rank=rank, shuffle=False, drop_last=True
            )
        )
        for rank in range(3)
    ]

    assert [len(shard) for shard in shards] == [3, 3, 3]
    assert sorted(value for shard in shards for value in shard) == list(range(9))


@pytest.mark.parametrize(
    "sampler_type, module",
    [(distributed.DistributedSampler, distributed), (nccl.DistributedSampler, nccl)],
)
def test_explicit_rank_zero_is_not_replaced_by_global_rank(
    sampler_type, module, monkeypatch
):
    monkeypatch.setattr(module, "get_rank", lambda: 1)
    sampler = sampler_type(list(range(4)), num_replicas=2, rank=0, shuffle=False)
    assert list(sampler) == [0, 2]


@pytest.mark.parametrize(
    "sampler_type", [distributed.DistributedSampler, nccl.DistributedSampler]
)
def test_invalid_rank_and_replica_count_are_rejected(sampler_type):
    dataset = list(range(4))
    with pytest.raises(ValueError, match="num_replicas"):
        sampler_type(dataset, num_replicas=0, rank=0)
    with pytest.raises(ValueError, match="rank"):
        sampler_type(dataset, num_replicas=2, rank=2)


def test_empty_dataset_has_empty_shards():
    for sampler_type in (distributed.DistributedSampler, nccl.DistributedSampler):
        sampler = sampler_type([], num_replicas=4, rank=3, shuffle=False)
        assert len(sampler) == 0
        assert list(sampler) == []
