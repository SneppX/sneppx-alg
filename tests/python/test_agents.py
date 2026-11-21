from SneppX_ALG import agents


def test_messages():
    conv = agents.Conversation()
    conv.add("system", "you are helpful")
    conv.add("user", "hi")
    assert len(conv) == 2
    assert conv.by_role("user")[0].content == "hi"
    assert conv.to_list()[0] == {"role": "system", "content": "you are helpful"}
    try:
        agents.Message("bogus", "x")
        assert False
    except ValueError:
        pass


def test_tool_registry():
    reg = agents.ToolRegistry()

    @reg.tool(name="add", description="add two ints")
    def add(text):
        a, b = text.split()
        return int(a) + int(b)

    @reg.tool()
    def upper(text):
        return text.upper()

    assert "add" in reg and "upper" in reg
    assert len(reg) == 2
    assert reg.call("add", "2 3") == 5
    assert reg.call("upper", "abc") == "ABC"
    assert sorted(reg.names()) == ["add", "upper"]


def test_parsing():
    text = "Thought: I should search\nAction: search\nAction Input: cats"
    p = agents.parse(text)
    assert p["thought"] == "I should search"
    assert p["action"] == "search"
    assert p["action_input"] == "cats"
    assert p["final"] is None

    done = agents.parse("Thought: done\nFinal Answer: 42")
    assert done["final"] == "42"
    assert done["action"] is None


def test_memory_trim():
    mem = agents.ConversationMemory(system="sys", max_turns=1)
    mem.add("user", "u1")
    mem.add("assistant", "a1")
    mem.add("user", "u2")
    mem.add("assistant", "a2")
    assert len(mem) == 3
    assert mem.messages()[0].role == "system"
    assert mem.messages()[-1].content == "a2"


def test_memory_clear():
    mem = agents.ConversationMemory(system="sys")
    mem.add("user", "a")
    mem.add("assistant", "b")
    mem.clear()
    assert len(mem) == 1
    assert mem.messages()[0].role == "system"


def test_agent_react_loop():
    reg = agents.ToolRegistry()
    calls = []

    @reg.tool(name="lookup")
    def lookup(text):
        calls.append(text)
        return {"cats": "42"}[text]

    scripted = [
        "Thought: look it up\nAction: lookup\nAction Input: cats",
        "Thought: I have it\nFinal Answer: 42",
    ]
    state = {"i": 0}

    def policy(conversation):
        out = scripted[state["i"]]
        state["i"] += 1
        return out

    agent = agents.Agent(policy, tools=reg)
    result = agent.run("how many cats?")
    assert result == "42"
    assert calls == ["cats"]
    assert agent.steps[0].action == "lookup"
    assert agent.steps[0].observation == "42"
    assert agent.steps[-1].final == "42"


def test_agent_unknown_tool():
    def policy(conversation):
        return "Action: nope\nAction Input: x"

    agent = agents.Agent(policy, max_steps=2)
    agent.run("do something")
    assert "unknown tool" in agent.steps[0].observation


def test_agent_max_steps():
    reg = agents.ToolRegistry()

    @reg.tool(name="echo")
    def echo(text):
        return text

    def loop_policy(conversation):
        return "Action: echo\nAction Input: x"

    agent = agents.Agent(loop_policy, tools=reg, max_steps=2)
    assert agent.run("x") == "max_steps exceeded"
