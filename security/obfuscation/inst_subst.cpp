#include "instruction_obfuscation_engine.h"
#include <algorithm>
#include <sstream>
#include <random>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>

/*
 * SNEPPX - Inst Subst
 *
 * WHAT
 *   Inst Subst.
 *
 * CONCEPT
 *   Provides the Inst Subst.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace SNEPPX {

SNEPPXObfSubst::SNEPPXObfSubst() : rng(std::random_device{}()) {}

void SNEPPXObfSubst::set_seed(uint64_t seed) {
    rng.seed(seed);
}

static SNEPPXObfInstruction make_mov(const std::string& result, const std::string& src) {
    SNEPPXObfInstruction m; m.type = SNEPPXObfInstType::MOV; m.result = result; m.operand1 = src; return m;
}

static SNEPPXObfInstruction make_add(const std::string& result, const std::string& op1, const std::string& op2) {
    SNEPPXObfInstruction a; a.type = SNEPPXObfInstType::ADD; a.result = result; a.operand1 = op1; a.operand2 = op2; return a;
}

static SNEPPXObfInstruction make_sub(const std::string& result, const std::string& op1, const std::string& op2) {
    SNEPPXObfInstruction s; s.type = SNEPPXObfInstType::SUB; s.result = result; s.operand1 = op1; s.operand2 = op2; return s;
}

static SNEPPXObfInstruction make_xor(const std::string& result, const std::string& op1, const std::string& op2) {
    SNEPPXObfInstruction x; x.type = SNEPPXObfInstType::XOR; x.result = result; x.operand1 = op1; x.operand2 = op2; return x;
}

static SNEPPXObfInstruction make_nand(const std::string& result, const std::string& op1, const std::string& op2) {
    SNEPPXObfInstruction n; n.type = SNEPPXObfInstType::NAND; n.result = result; n.operand1 = op1; n.operand2 = op2; return n;
}

static SNEPPXObfInstruction make_push(const std::string& src) {
    SNEPPXObfInstruction p; p.type = SNEPPXObfInstType::PUSH; p.operand1 = src; return p;
}

static SNEPPXObfInstruction make_pop(const std::string& dst) {
    SNEPPXObfInstruction p; p.type = SNEPPXObfInstType::POP; p.result = dst; return p;
}

static SNEPPXObfInstruction make_and(const std::string& result, const std::string& op1, const std::string& op2) {
    SNEPPXObfInstruction a; a.type = SNEPPXObfInstType::AND; a.result = result; a.operand1 = op1; a.operand2 = op2; return a;
}

static SNEPPXObfInstruction make_or(const std::string& result, const std::string& op1, const std::string& op2) {
    SNEPPXObfInstruction o; o.type = SNEPPXObfInstType::OR; o.result = result; o.operand1 = op1; o.operand2 = op2; return o;
}

static SNEPPXObfInstruction make_mul_inst(const SNEPPXObfInstruction& inst) {
    SNEPPXObfInstruction m; m.type = SNEPPXObfInstType::MUL; m.result = inst.result; m.operand1 = inst.operand1; m.operand2 = inst.operand2; return m;
}

static std::string temp_name(const std::string& base, int idx) {
    char buf[64]; snprintf(buf, sizeof(buf), "_t%d_%s", idx, base.c_str()); return std::string(buf);
}

bool SNEPPXObfSubst::choose_substitution() {
    return (rng() % 2) == 0;
}

int SNEPPXObfSubst::rand_int(int min, int max) {
    return min + (rng() % (max - min + 1));
}

SNEPPXObfInstruction SNEPPXObfSubst::make_lea_add(const SNEPPXObfInstruction& inst) {
    return substitute_add_inst(inst);
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::make_neg_sub_add(const SNEPPXObfInstruction& inst) {
    return substitute_sub_inst(inst);
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::make_mul_shift_add(const SNEPPXObfInstruction& inst) {
    return substitute_mul_inst(inst);
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::make_nand_and(const SNEPPXObfInstruction& inst) {
    return substitute_and_inst(inst);
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::make_nand_or(const SNEPPXObfInstruction& inst) {
    return substitute_or_inst(inst);
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::make_nand_xor(const SNEPPXObfInstruction& inst) {
    return substitute_xor_inst(inst);
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::make_sub_cmp(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    seq.push_back(make_sub(inst.result, inst.operand1, inst.operand2));
    return seq;
}

SNEPPXObfInstruction SNEPPXObfSubst::substitute_add_inst(const SNEPPXObfInstruction& inst) {
    int choice = rng() % 3;
    if (choice == 0) {
        SNEPPXObfInstruction lea;
        lea.type = SNEPPXObfInstType::LEA;
        lea.result = inst.result;
        lea.operand1 = "[" + inst.operand1 + " + " + inst.operand2 + "]";
        return lea;
    } else if (choice == 1) {
        SNEPPXObfInstruction x; x.type = SNEPPXObfInstType::XOR; x.result = inst.result;
        x.operand1 = inst.operand1; x.operand2 = inst.operand2;
        return x;
    } else {
        SNEPPXObfInstruction s; s.type = SNEPPXObfInstType::SUB; s.result = inst.result;
        s.operand1 = inst.operand1;
        s.operand2 = "-(" + inst.operand2 + ")";
        return s;
    }
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_sub_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    SNEPPXObfInstruction neg;
    neg.type = SNEPPXObfInstType::NEG;
    neg.operand1 = inst.operand2;
    neg.result = temp_name(inst.operand2, rng() % 1000);
    seq.push_back(neg);
    seq.push_back(make_add(inst.result, inst.operand1, neg.result));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_mul_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    int k = rand_int(2, 5);
    std::string acc = temp_name("mul", rng() % 1000);
    seq.push_back(make_mov(acc, inst.operand1));
    for (int i = 1; i < k; i++) {
        std::string tmp = temp_name("m", rng() % 1000);
        seq.push_back(make_add(tmp, acc, inst.operand1));
        acc = tmp;
    }
    seq.push_back(make_mov(inst.result, acc));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_div_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string t = temp_name("div", rng() % 1000);
    seq.push_back(make_add(t, inst.operand1, inst.operand1));
    std::string t2 = temp_name("d", rng() % 1000);
    seq.push_back(make_add(t2, t, inst.operand1));
    if (inst.operand2 == "2") {
        SNEPPXObfInstruction s; s.type = SNEPPXObfInstType::SHL;
        s.result = t2; s.operand1 = t2; s.operand2 = "2";
        seq.push_back(s);
    }
    seq.push_back(make_mov(inst.result, t2));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_and_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string t = temp_name("and", rng() % 1000);
    seq.push_back(make_nand(t, inst.operand1, inst.operand2));
    seq.push_back(make_nand(inst.result, t, t));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_or_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string t1 = temp_name("or", rng() % 1000);
    std::string t2 = temp_name("r", rng() % 1000);
    seq.push_back(make_nand(t1, inst.operand1, inst.operand1));
    seq.push_back(make_nand(t2, inst.operand2, inst.operand2));
    seq.push_back(make_nand(inst.result, t1, t2));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_xor_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string nab = temp_name("x", rng() % 1000);
    std::string na = temp_name("y", rng() % 1000);
    std::string nb = temp_name("z", rng() % 1000);
    seq.push_back(make_nand(nab, inst.operand1, inst.operand2));
    seq.push_back(make_nand(na, inst.operand1, nab));
    seq.push_back(make_nand(nb, inst.operand2, nab));
    seq.push_back(make_nand(inst.result, na, nb));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_not_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string ones = temp_name("not", rng() % 1000);
    SNEPPXObfInstruction mov_neg1;
    mov_neg1.type = SNEPPXObfInstType::MOV;
    mov_neg1.result = ones;
    mov_neg1.operand1 = "-1";
    seq.push_back(mov_neg1);
    seq.push_back(make_xor(inst.result, inst.operand1, ones));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_neg_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string not_v = temp_name("neg", rng() % 1000);
    auto not_seq = substitute_not_inst(inst);
    for (auto& n : not_seq) {
        if (n.type == SNEPPXObfInstType::XOR) {
            SNEPPXObfInstruction nx; nx.type = SNEPPXObfInstType::XOR;
            nx.result = not_v; nx.operand1 = n.operand1; nx.operand2 = n.operand2;
            seq.push_back(nx);
        } else {
            seq.push_back(n);
        }
    }
    seq.push_back(make_add(inst.result, not_v, "1"));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_shl_inst(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string acc = temp_name("shl", rng() % 1000);
    seq.push_back(make_mov(acc, inst.operand1));
    int shift = 1;
    try { shift = std::stoi(inst.operand2); } catch (...) { shift = 1; }
    for (int i = 0; i < shift; i++) {
        std::string t = temp_name("s", rng() % 1000);
        seq.push_back(make_add(t, acc, acc));
        acc = t;
    }
    seq.push_back(make_mov(inst.result, acc));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_add_lea_scaled(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    int scale = 1 << (rng() % 4);
    char scale_buf[16]; snprintf(scale_buf, sizeof(scale_buf), "%d", scale);
    SNEPPXObfInstruction lea;
    lea.type = SNEPPXObfInstType::LEA;
    lea.result = inst.result;
    lea.operand1 = inst.operand1;
    lea.operand2 = "[" + inst.operand2 + " * " + std::string(scale_buf) + "]";
    seq.push_back(lea);
    if (scale > 1) {
        std::string rem = temp_name("rem", rng() % 1000);
        seq.push_back(make_sub(rem, lea.result, inst.operand1));
        SNEPPXObfInstruction lea2;
        lea2.type = SNEPPXObfInstType::LEA;
        lea2.result = inst.result;
        lea2.operand1 = rem;
        lea2.operand2 = inst.operand2;
        seq.push_back(lea2);
    }
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_sub_neg_adc(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string neg_d = temp_name("n", rng() % 1000);
    SNEPPXObfInstruction neg;
    neg.type = SNEPPXObfInstType::NEG;
    neg.result = neg_d;
    neg.operand1 = inst.operand2;
    seq.push_back(neg);
    std::string carry = temp_name("c", rng() % 1000);
    seq.push_back(make_mov(carry, "0"));
    std::string not_c = temp_name("nc", rng() % 1000);
    auto not_seq = substitute_not_inst({SNEPPXObfInstType::NOT, "0", "", carry});
    for (auto& n : not_seq) {
        SNEPPXObfInstruction ad;
        ad.type = n.type; ad.result = not_c; ad.operand1 = n.operand1; ad.operand2 = n.operand2;
        seq.push_back(ad);
    }
    seq.push_back(make_add(inst.result, inst.operand1, neg_d));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_mul_karatsuba(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string a_hi = temp_name("ahi", rng() % 1000);
    std::string a_lo = temp_name("alo", rng() % 1000);
    std::string b_hi = temp_name("bhi", rng() % 1000);
    std::string b_lo = temp_name("blo", rng() % 1000);
    SNEPPXObfInstruction mov_ahi; mov_ahi.type = SNEPPXObfInstType::MOV; mov_ahi.result = a_hi; mov_ahi.operand1 = inst.operand1 + "_hi"; seq.push_back(mov_ahi);
    SNEPPXObfInstruction mov_alo; mov_alo.type = SNEPPXObfInstType::MOV; mov_alo.result = a_lo; mov_alo.operand1 = inst.operand1; seq.push_back(mov_alo);
    SNEPPXObfInstruction mov_bhi; mov_bhi.type = SNEPPXObfInstType::MOV; mov_bhi.result = b_hi; mov_bhi.operand1 = inst.operand2 + "_hi"; seq.push_back(mov_bhi);
    SNEPPXObfInstruction mov_blo; mov_blo.type = SNEPPXObfInstType::MOV; mov_blo.result = b_lo; mov_blo.operand1 = inst.operand2; seq.push_back(mov_blo);
    std::string z0 = temp_name("z0", rng() % 1000);
    std::string z1 = temp_name("z1", rng() % 1000);
    std::string z2 = temp_name("z2", rng() % 1000);
    std::string p1 = temp_name("p1", rng() % 1000);
    std::string p2 = temp_name("p2", rng() % 1000);
    seq.push_back(make_mul_inst({SNEPPXObfInstType::MUL, a_lo, b_lo, z0}));
    seq.push_back(make_mul_inst({SNEPPXObfInstType::MUL, a_hi, b_hi, z2}));
    seq.push_back(make_add(p1, a_lo, a_hi));
    seq.push_back(make_add(p2, b_lo, b_hi));
    seq.push_back(make_mul_inst({SNEPPXObfInstType::MUL, p1, p2, z1}));
    seq.push_back(make_sub(z1, z1, z0));
    seq.push_back(make_sub(z1, z1, z2));
    seq.push_back(make_add(inst.result, z0, z1));
    seq.push_back(make_add(inst.result, inst.result, z2));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_and_nand_variant(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string t1 = temp_name("av", rng() % 1000);
    std::string t2 = temp_name("av2", rng() % 1000);
    seq.push_back(make_nand(t1, inst.operand1, inst.operand1));
    seq.push_back(make_nand(t2, inst.operand2, inst.operand2));
    seq.push_back(make_nand(t1, t1, t2));
    seq.push_back(make_nand(t2, t1, t1));
    seq.push_back(make_nand(inst.result, t2, t2));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_or_nand_variant(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string t1 = temp_name("ov", rng() % 1000);
    std::string t2 = temp_name("ov2", rng() % 1000);
    std::string t3 = temp_name("ov3", rng() % 1000);
    seq.push_back(make_nand(t1, inst.operand1, inst.operand2));
    seq.push_back(make_nand(t2, inst.operand1, t1));
    seq.push_back(make_nand(t3, inst.operand2, t1));
    seq.push_back(make_nand(inst.result, t2, t3));
    return seq;
}

std::vector<SNEPPXObfInstruction> SNEPPXObfSubst::substitute_xor_nand_variant(const SNEPPXObfInstruction& inst) {
    std::vector<SNEPPXObfInstruction> seq;
    std::string n1 = temp_name("xv", rng() % 1000);
    std::string n2 = temp_name("xv2", rng() % 1000);
    std::string n3 = temp_name("xv3", rng() % 1000);
    std::string n4 = temp_name("xv4", rng() % 1000);
    seq.push_back(make_nand(n1, inst.operand1, inst.operand1));
    seq.push_back(make_nand(n2, inst.operand2, inst.operand2));
    seq.push_back(make_nand(n3, n1, n2));
    seq.push_back(make_nand(n1, inst.operand1, inst.operand2));
    seq.push_back(make_nand(n4, n3, n1));
    seq.push_back(make_nand(inst.result, n4, n4));
    return seq;
}

void SNEPPXObfSubst::insert_junk(SNEPPXObfBlock& block) {
    std::vector<SNEPPXObfInstruction> new_insts;
    for (auto& inst : block.instructions) {
        new_insts.push_back(inst);
        if (choose_substitution()) {
            int junk_type = rng() % 3;
            if (junk_type == 0) {
                SNEPPXObfInstruction j;
                j.type = SNEPPXObfInstType::NOP;
                new_insts.push_back(j);
            } else if (junk_type == 1) {
                new_insts.push_back(make_mov(inst.result, inst.operand1));
            } else {
                std::string dummy = temp_name("junk", rng() % 1000);
                new_insts.push_back(make_xor(dummy, inst.operand1, inst.operand2));
            }
        }
    }
    block.instructions = new_insts;
}

void SNEPPXObfSubst::insert_junk_extended(SNEPPXObfBlock& block) {
    std::vector<SNEPPXObfInstruction> new_insts;
    for (auto& inst : block.instructions) {
        new_insts.push_back(inst);
        if (choose_substitution()) {
            int junk_type = rng() % 8;
            switch (junk_type) {
                case 0: {
                    SNEPPXObfInstruction j; j.type = SNEPPXObfInstType::NOP; new_insts.push_back(j);
                    break;
                }
                case 1: {
                    new_insts.push_back(make_mov(inst.result, inst.operand1));
                    break;
                }
                case 2: {
                    std::string dummy = temp_name("dead", rng() % 1000);
                    new_insts.push_back(make_mov(dummy, "0"));
                    break;
                }
                case 3: {
                    new_insts.push_back(make_add(temp_name("zadd", rng() % 1000), inst.operand1, "0"));
                    break;
                }
                case 4: {
                    new_insts.push_back(make_xor(temp_name("zxor", rng() % 1000), inst.operand1, "0"));
                    break;
                }
                case 5: {
                    new_insts.push_back(make_push(inst.operand1));
                    new_insts.push_back(make_pop(inst.operand1));
                    break;
                }
                case 6: {
                    std::string r_src = temp_name("r", rng() % 1000);
                    std::string r_dst = temp_name("s", rng() % 1000);
                    new_insts.push_back(make_mov(r_src, inst.operand1));
                    new_insts.push_back(make_mov(r_dst, r_src));
                    new_insts.push_back(make_add(r_dst, r_dst, "0"));
                    break;
                }
                case 7: {
                    std::string t = temp_name("opt", rng() % 1000);
                    new_insts.push_back(make_mov(t, "0"));
                    if (inst.type == SNEPPXObfInstType::ADD) {
                        new_insts.push_back(make_add(t, inst.operand1, inst.operand2));
                    } else if (inst.type == SNEPPXObfInstType::SUB) {
                        new_insts.push_back(make_sub(t, inst.operand1, inst.operand2));
                    } else if (inst.type == SNEPPXObfInstType::XOR) {
                        new_insts.push_back(make_xor(t, inst.operand1, inst.operand2));
                    } else {
                        new_insts.push_back(make_and(t, inst.operand1, inst.operand2));
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }
    block.instructions = new_insts;
}

void SNEPPXObfSubst::rename_registers_block(SNEPPXObfBlock& block, int& next_temp) {
    std::unordered_map<std::string, std::string> rename_map;
    std::vector<std::string> used_vars;
    for (auto& inst : block.instructions) {
        if (!inst.result.empty() && inst.result[0] != '_') {
            if (!rename_map.count(inst.result)) {
                std::string new_name = temp_name("rn", next_temp++);
                rename_map[inst.result] = new_name;
                used_vars.push_back(inst.result);
            }
        }
        if (!inst.operand1.empty() && inst.operand1[0] != '_' && inst.operand1.find('[') == std::string::npos) {
            if (!rename_map.count(inst.operand1)) {
                std::string new_name = temp_name("rn", next_temp++);
                rename_map[inst.operand1] = new_name;
                used_vars.push_back(inst.operand1);
            }
        }
        if (!inst.operand2.empty() && inst.operand2[0] != '_' && inst.operand2.find('[') == std::string::npos) {
            if (!rename_map.count(inst.operand2)) {
                std::string new_name = temp_name("rn", next_temp++);
                rename_map[inst.operand2] = new_name;
                used_vars.push_back(inst.operand2);
            }
        }
    }
    for (auto& inst : block.instructions) {
        if (!inst.result.empty() && rename_map.count(inst.result)) inst.result = rename_map[inst.result];
        if (!inst.operand1.empty() && rename_map.count(inst.operand1)) inst.operand1 = rename_map[inst.operand1];
        if (!inst.operand2.empty() && rename_map.count(inst.operand2)) inst.operand2 = rename_map[inst.operand2];
    }
}

void SNEPPXObfSubst::rename_registers_cfg(SNEPPXObfCFG& cfg) {
    int next_temp = 0;
    for (auto& pair : cfg.blocks) {
        rename_registers_block(*pair.second, next_temp);
    }
}

void SNEPPXObfSubst::substitute_add(SNEPPXObfBlock& block) {
    (void)block;
}

void SNEPPXObfSubst::substitute_logic(SNEPPXObfBlock& block) {
    for (size_t i = 0; i < block.instructions.size(); i++) {
        auto& inst = block.instructions[i];
        std::vector<SNEPPXObfInstruction> seq;
        if (inst.type == SNEPPXObfInstType::ADD) {
            int mode = rng() % 2;
            if (mode == 0) {
                SNEPPXObfInstruction r = substitute_add_inst(inst);
                inst = r;
            } else {
                seq = substitute_add_lea_scaled(inst);
            }
        } else if (inst.type == SNEPPXObfInstType::SUB) {
            int mode = rng() % 2;
            if (mode == 0) {
                seq = substitute_sub_inst(inst);
            } else {
                seq = substitute_sub_neg_adc(inst);
            }
        } else if (inst.type == SNEPPXObfInstType::MUL) {
            int mode = rng() % 2;
            if (mode == 0) {
                seq = substitute_mul_inst(inst);
            } else {
                seq = substitute_mul_karatsuba(inst);
            }
        } else if (inst.type == SNEPPXObfInstType::DIV) {
            seq = substitute_div_inst(inst);
        } else if (inst.type == SNEPPXObfInstType::AND) {
            int mode = rng() % 2;
            if (mode == 0) {
                seq = substitute_and_inst(inst);
            } else {
                seq = substitute_and_nand_variant(inst);
            }
        } else if (inst.type == SNEPPXObfInstType::OR) {
            int mode = rng() % 2;
            if (mode == 0) {
                seq = substitute_or_inst(inst);
            } else {
                seq = substitute_or_nand_variant(inst);
            }
        } else if (inst.type == SNEPPXObfInstType::XOR) {
            int mode = rng() % 2;
            if (mode == 0) {
                seq = substitute_xor_inst(inst);
            } else {
                seq = substitute_xor_nand_variant(inst);
            }
        } else if (inst.type == SNEPPXObfInstType::NOT) {
            seq = substitute_not_inst(inst);
        } else if (inst.type == SNEPPXObfInstType::NEG) {
            seq = substitute_neg_inst(inst);
        } else if (inst.type == SNEPPXObfInstType::SHL) {
            seq = substitute_shl_inst(inst);
        }
        if (!seq.empty()) {
            inst = seq[0];
            for (size_t j = 1; j < seq.size(); j++) {
                block.instructions.insert(block.instructions.begin() + (int)(i + j), seq[j]);
            }
            i += seq.size() - 1;
        }
    }
}

void SNEPPXObfSubst::substitute_compare(SNEPPXObfBlock& block) {
    for (auto& inst : block.instructions) {
        if (inst.type == SNEPPXObfInstType::CMP && choose_substitution()) {
            auto seq = make_sub_cmp(inst);
            inst = seq[0];
        }
    }
}

void SNEPPXObfSubst::substitute_all(SNEPPXObfBlock& block) {
    substitute_logic(block);
    substitute_compare(block);
    insert_junk(block);
    insert_junk_extended(block);
}

void SNEPPXObfSubst::substitute_all_blocks(SNEPPXObfCFG& cfg) {
    int next_temp = 1000;
    for (auto& pair : cfg.blocks) {
        substitute_all(*pair.second);
        rename_registers_block(*pair.second, next_temp);
    }
}

} // namespace SNEPPX
