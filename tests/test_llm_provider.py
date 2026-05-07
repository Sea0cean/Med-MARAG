# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: tests/test_llm_provider.py
Author: SeaOcean
Create Date: 2026-03-18
Description：LLM Provider 测试
-------------------------------------------------
"""
from __future__ import annotations

from config import Config
from utils.plantuml_utils import PlantUMLUtils
from llm.provider import build_llm_runtime
from workflow.ocean_graph import PipelineConfig, run_modeller_pipeline
from scripts.run_full_experiment import get_group_names, get_mas_group_specs, load_cases


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeChatOpenAI:
    def __init__(self, **_: object) -> None:
        pass

    def invoke(self, messages: list[tuple[str, str]]) -> FakeResponse:
        system_prompt = messages[0][1]
        user_prompt = messages[1][1]
        if "类图" in user_prompt and "design_elements" in user_prompt:
            return FakeResponse(
                '{"plantuml_code":"说明文字\\n@startuml\\nclass Patient {\\n  +patientId: String\\n}\\nclass Appointment {\\n  +createAppointment()\\n}\\nPatient \\"1\\" -- \\"*\\" Appointment : creates\\n@enduml\\n额外说明",'
                '"design_elements":["Patient","Appointment"],'
                '"mapping_summary":"完成实体与关系映射"}'
            )
        if "用例图" in user_prompt and "design_elements" in user_prompt:
            return FakeResponse(
                '{"plantuml_code":"@startuml\\nleft to right direction\\nactor \\"患者\\" as Patient\\nrectangle \\"Med-MARAG\\" {\\n  usecase \\"提交预约申请\\" as UC001\\n  usecase \\"接收确认通知\\" as UC002\\n}\\nPatient --> UC001\\nPatient --> UC002\\n@enduml",'
                '"design_elements":["患者","提交预约申请","接收确认通知"],'
                '"mapping_summary":"输出用例图"}'
            )
        if "序列图" in user_prompt and "design_elements" in user_prompt:
            return FakeResponse(
                '{"plantuml_code":"@startuml\\nactor \\"患者\\" as Patient\\nparticipant \\"医疗系统\\" as System\\nparticipant \\"通知服务\\" as Notify\\nPatient -> System: 提交预约申请\\nSystem -> System: 创建预约记录\\nSystem -> Notify: 发送确认通知\\nNotify --> System: 返回发送结果\\nSystem --> Patient: 返回预约成功信息\\n@enduml",'
                '"design_elements":["患者","医疗系统","通知服务","预约流程"],'
                '"mapping_summary":"输出序列图"}'
            )
        if "PlantUML:" in user_prompt:
            return FakeResponse(
                '{"scores":{"accuracy":90,"completeness":90,"clarity":90,"compliance":90,"overall":90},'
                '"issues":[],"suggestions":["保持当前结构"]}'
            )
        return FakeResponse(
            '<thinking>识别到参与者为患者，核心实体为预约记录与通知。'
            '业务主流程完整，但需保持需求表述规范化。</thinking>'
            '{"actors":["患者","系统"],"entities":["Appointment","Notification"],'
            '"business_flow":["患者提交预约请求","系统生成预约记录","系统发送确认通知"],'
            '"missing_info":[],"needs_clarification":false,"clarification_questions":[],"ears_requirement":"IF 患者预约挂号成功 THEN 系统应发送确认短信通知 ENDIF",'
            '"summary":"规范化需求输出","risks":["需校验通知渠道稳定性"],'
            '"use_case":{"name":"发送确认通知","goal":"患者收到预约确认"}}'
        )


def test_deepseek_legacy_env_is_supported(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")

    assert Config.llm_enabled("deepseek") is True
    assert Config.get_llm_api_key("deepseek") == "deepseek-test-key"
    assert Config.get_llm_model("deepseek") == "deepseek-chat"


def test_openai_provider_uses_openai_envs(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")

    assert Config.llm_enabled("openai") is True
    assert Config.get_llm_api_key("openai") == "openai-test-key"
    assert Config.get_llm_model("openai") == "gpt-4.1-mini"


def test_invalid_provider_and_missing_key_fallback(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    invalid = build_llm_runtime(provider="invalid-provider")
    assert invalid.effective_provider == "offline"
    assert invalid.status == "fallback:invalid_provider"

    missing_key = build_llm_runtime(provider="deepseek")
    assert missing_key.effective_provider == "offline"
    assert missing_key.status == "fallback:missing_api_key"


def test_pipeline_records_runtime_metadata_for_mocked_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setattr("llm.provider.ChatOpenAI", FakeChatOpenAI)

    state = run_modeller_pipeline(
        "当患者预约挂号成功时，系统应发送确认短信给患者",
        config=PipelineConfig(provider="deepseek", max_iterations=2),
    )

    assert state["LLM_Provider"] == "deepseek"
    assert state["LLM_Runtime_Status"] == "enabled"
    assert state["LLM_Enabled"] is True
    assert state["Requirement_Items"][0]["llm_used"] is True
    assert state["Review_Report"]["review_source"] == "llm"
    assert state["UML_Artifacts"]["use_case_diagram_source"] == "llm"
    assert state["UML_Artifacts"]["class_diagram_source"] == "llm"
    assert state["UML_Artifacts"]["sequence_diagram_source"] == "llm"
    assert state["UML_Artifacts"]["llm_sandbox_issues"] == []
    assert state["RA_Artifacts"]["RA-1"]["source"] == "llm"
    assert state["RA_Artifacts"]["RA-5"]["source"] == "llm"


def test_pipeline_records_runtime_metadata_for_mocked_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr("llm.provider.ChatOpenAI", FakeChatOpenAI)

    state = run_modeller_pipeline(
        "当患者预约挂号成功时，系统应发送确认短信给患者",
        config=PipelineConfig(provider="openai", max_iterations=2),
    )

    assert state["LLM_Provider"] == "openai"
    assert state["LLM_Runtime_Status"] == "enabled"
    assert state["LLM_Enabled"] is True
    assert state["UML_Artifacts"]["use_case_diagram_source"] == "llm"
    assert state["UML_Artifacts"]["class_diagram"].startswith("@startuml")
    assert state["UML_Artifacts"]["sequence_diagram_source"] == "llm"
    assert "createAppointment()" in state["UML_Artifacts"]["class_diagram"]


def test_pipeline_with_llm_can_disable_rag(monkeypatch):
    from rag.knowledge_base import knowledge_base

    def _unexpected_query(*args, **kwargs):
        raise AssertionError("knowledge_base.query should not be called when enable_rag=False")

    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr("llm.provider.ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(knowledge_base, "query", _unexpected_query)

    state = run_modeller_pipeline(
        "当患者预约挂号成功时，系统应发送确认短信给患者",
        config=PipelineConfig(provider="openai", enable_rag=False, max_iterations=2),
    )

    assert state["RAG_Enabled"] is False
    assert state["LLM_Provider"] == "openai"
    assert state["Review_Report"]["review_source"] == "llm"
    assert state["Knowledge_Context"] == []


def test_pipeline_continues_and_exposes_suggestions_when_analyst_requests_clarification(monkeypatch):
    class ClarificationChatOpenAI:
        def __init__(self, **_: object) -> None:
            pass

        def invoke(self, messages: list[tuple[str, str]]) -> FakeResponse:
            system_prompt = messages[0][1]
            if "你是 UML 建模专家" in system_prompt:
                return FakeResponse("@startuml\nclass Patient\n@enduml")
            return FakeResponse(
                '<thinking>已识别支付场景，但未发现失败或超时处理描述，需要补充异常路径。</thinking>'
                '{"actors":["患者","系统"],"entities":["BillingRecord"],'
                '"business_flow":["患者发起支付","系统处理支付结果"],'
                '"missing_info":["缺少支付失败或超时处理"],'
                '"needs_clarification":true,'
                '"clarification_questions":["请补充支付失败、超时或取消时系统应如何处理。"],'
                '"ears_requirement":"",'
                '"summary":"当前需求缺少支付异常路径。","risks":["异常处理缺失将导致需求不可验证"],'
                '"use_case":{"name":"处理支付","goal":"系统完成支付处理"}}'
            )

    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr("llm.provider.ChatOpenAI", ClarificationChatOpenAI)

    state = run_modeller_pipeline(
        "当患者完成支付时，系统应更新支付状态",
        config=PipelineConfig(provider="openai", max_iterations=2),
    )

    assert state["Status"] in {"REVIEW_PASSED", "HUMAN_REVIEW_REQUIRED", "NEEDS_REVISION", "NEEDS_REQUIREMENT_REVISION"}
    assert state["Clarification_Questions"] == ["建议补充：支付失败/超时/取消处理规则。"]
    # Pipeline should still produce UML artifacts even if suggestions exist.
    assert state["UML_Artifacts"]["class_diagram"].startswith("@startuml")
    assert state["Analyst_Thinking"]


def test_plantuml_sandbox_preprocess_extracts_core_code():
    raw = """
这里是类图说明
```plantuml
@startuml
class Patient {
  +patientId: String
}
Patient "1" -- "*" Appointment : creates
@enduml
```
补充说明
"""
    result = PlantUMLUtils.sandbox_preprocess(raw)

    assert result["code"].startswith("@startuml")
    assert result["code"].endswith("@enduml")
    assert "这里是类图说明" not in result["code"]
    assert result["issues"] == []


def test_experiment_groups_keep_current_default_and_support_optional_mas_without_rag():
    assert get_group_names() == [
        "Only LLM",
        "LLM with RAG",
        "MAS + RAG (max_iter=1)",
        "MAS + RAG (max_iter=3)",
        "MAS + RAG (max_iter=5)",
    ]
    assert get_group_names(include_mas_without_rag=True) == [
        "Only LLM",
        "LLM with RAG",
        "LLM with MAS (max_iter=1)",
        "LLM with MAS (max_iter=3)",
        "LLM with MAS (max_iter=5)",
        "MAS + RAG (max_iter=1)",
        "MAS + RAG (max_iter=3)",
        "MAS + RAG (max_iter=5)",
    ]
    assert get_mas_group_specs() == [
        (1, "MAS + RAG (max_iter=1)", True),
        (3, "MAS + RAG (max_iter=3)", True),
        (5, "MAS + RAG (max_iter=5)", True),
    ]
    assert get_group_names(include_mas_without_rag=True, mas_without_rag_iters=[1]) == [
        "Only LLM",
        "LLM with RAG",
        "LLM with MAS (max_iter=1)",
        "MAS + RAG (max_iter=1)",
        "MAS + RAG (max_iter=3)",
        "MAS + RAG (max_iter=5)",
    ]
    assert get_mas_group_specs(include_mas_without_rag=True, mas_without_rag_iters=[1]) == [
        (1, "LLM with MAS (max_iter=1)", False),
        (1, "MAS + RAG (max_iter=1)", True),
        (3, "MAS + RAG (max_iter=3)", True),
        (5, "MAS + RAG (max_iter=5)", True),
    ]


def test_load_cases_uses_random_2_to_3_ratio_when_sampling_less_than_full_dataset(tmp_path):
    import pandas as pd

    rows = []
    for index in range(1, 11):
        rows.append({"id": f"B{index:03d}", "category": "基础交互", "requirement": f"基础需求{index}"})
    for index in range(1, 16):
        rows.append({"id": f"C{index:03d}", "category": "复杂业务", "requirement": f"复杂需求{index}"})

    csv_path = tmp_path / "requirements_input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    sampled = load_cases(csv_path, max_cases=10, sample_seed=12345)

    assert len(sampled) == 10
    assert sum(1 for item in sampled if item["category"] == "基础交互") == 4
    assert sum(1 for item in sampled if item["category"] == "复杂业务") == 6
