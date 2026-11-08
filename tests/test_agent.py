"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from types import SimpleNamespace

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent, run_with_approvals

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_waits_for_approval(tmp_path, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        result = agent.run_sync("今日学んだことを記録して")

    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.approvals] == ["save_learning_log"]
    assert not (tmp_path / "learning_log").exists()


def test_run_with_approvals_resumes_with_history_and_each_decision(monkeypatch):
    requests = DeferredToolRequests(
        approvals=[
            ToolCallPart(
                tool_name="save_learning_log",
                args={"content": "first"},
                tool_call_id="call-1",
            ),
            ToolCallPart(
                tool_name="save_learning_log",
                args={"content": "second"},
                tool_call_id="call-2",
            ),
        ]
    )
    history = [object()]
    final_result = SimpleNamespace(output="done")
    calls = []

    class Target:
        def run_sync(self, *args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                return SimpleNamespace(output=requests, all_messages=lambda: history)
            return final_result

    answers = iter(["y", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    result = run_with_approvals(Target(), "question", model="test-model")

    assert result is final_result
    assert calls[0][0] == ("question",)
    resumed_kwargs = calls[1][1]
    assert resumed_kwargs["message_history"] is history
    assert resumed_kwargs["deferred_tool_results"].approvals == {
        "call-1": True,
        "call-2": False,
    }
