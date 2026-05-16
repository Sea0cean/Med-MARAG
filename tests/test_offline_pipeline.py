# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: tests/test_offline_pipeline.py
Author: SeaOcean
Create Date: 2026-03-13
Description：离线流水线测试
-------------------------------------------------
"""
from workflow.ocean_graph import PipelineConfig, _review_step, run_modeller_pipeline
from utils.requirement_utils import RequirementUtils
from agents.review_agent import ReviewAgent


def test_requirement_to_ears():
    requirement = "当患者登录系统时，系统应验证其身份并显示个人健康记录"
    ears = RequirementUtils.convert_to_ears(requirement)
    assert "IF 患者登录系统" in ears
    assert "THEN 系统应验证其身份" in ears
    assert "AND 系统应显示个人健康记录" in ears


def test_requirement_validation_score_is_not_always_full_mark():
    strong_requirement = "当患者预约挂号成功后，系统应发送确认短信并更新挂号状态，同时记录审计日志。"
    weak_requirement = "系统进行相关处理。"

    strong_validation = RequirementUtils.validate_requirement(strong_requirement)
    weak_validation = RequirementUtils.validate_requirement(weak_requirement)

    assert strong_validation["score"] < 100
    assert strong_validation["score"] >= 75
    assert weak_validation["score"] < strong_validation["score"]
    assert weak_validation["issues"]


def test_offline_pipeline_outputs_complete_artifacts():
    state = run_modeller_pipeline(
        "当患者预约挂号成功时，系统应发送确认短信给患者",
        config=PipelineConfig(provider="offline", max_iterations=2),
    )
    assert state["Status"] in {"REVIEW_PASSED", "HUMAN_REVIEW_REQUIRED"}
    assert state["LLM_Provider"] == "offline"
    assert state["LLM_Runtime_Status"] == "offline"
    assert state["Requirement_Items"]
    assert state["Use_Cases"]
    assert state["UML_Artifacts"]["class_diagram"].startswith("@startuml")
    assert "+createAppointment()" in state["UML_Artifacts"]["class_diagram"]
    assert "+sendNotification()" in state["UML_Artifacts"]["class_diagram"]
    assert state["Test_Cases"]
    assert state["Traceability_Matrix"]
    assert "Knowledge_Context" in state
    assert 'Patient "1" -- "*" Appointment : creates' in state["RA_Artifacts"]["RA-4"]["plantuml"]
    assert state["RA_Artifacts"]["RA-5"]["plantuml"].startswith("@startuml")
    assert "use_cases" not in state["RA_Artifacts"]["RA-5"]


def test_split_requirements_breaks_complex_emergency_flow_into_multiple_items():
    requirement = (
        "当急诊患者到达医院时，分诊护士需要在系统中录入患者的基本信息，以及心率、血压、血氧等生命体征数据。"
        "系统会基于这些指标，自动计算患者的初步急诊分级，一般划分为1至4级。"
        "对于被评定为3级或4级的普通急诊患者，系统会引导其前往自助挂号机或者人工窗口完成挂号操作。"
        "在挂号过程中，患者可以选择微信、支付宝以及医保卡等方式进行支付。"
        "支付完成后，系统会生成对应的急诊挂号流水号，并且将患者信息推送至相关科室的医生工作站排队系统。"
        "医生在工作站接诊患者时，可以查看其分诊记录，并根据实际情况开具电子处方。"
        "通过这一流程，患者的就诊信息可以在系统中实现连续记录与统一管理。"
        "对于被评定为1级或2级的危重症患者，处理流程会有所不同。"
        "此时护士需要立即安排抢救，系统应当支持跳过缴费环节，直接为患者建立临时急诊档案。"
        "等抢救结束之后，系统还需要支持补缴挂号费用以及抢救费用，以保证业务流程的完整性。"
        "在数据管理方面，所有进入急诊系统的患者档案与处方数据，都需要在每天零点进行自动同步，并上传至医院核心数据仓库进行统一备份。"
        "如果在同步过程中出现网络超时或者连接失败的情况，系统需要记录相关错误日志，同时通过短信方式通知运维人员进行处理。"
    )

    parts = RequirementUtils.split_requirements(requirement)

    assert len(parts) >= 4
    assert parts[0].startswith("当急诊患者到达医院时")
    assert any(part.startswith("对于被评定为3级或4级") for part in parts)
    assert any(part.startswith("对于被评定为1级或2级") for part in parts)
    assert any(part.startswith("在数据管理方面") for part in parts)


def test_offline_pipeline_generates_multiple_use_cases_for_complex_emergency_flow():
    requirement = (
        "当急诊患者到达医院时，分诊护士需要在系统中录入患者的基本信息，以及心率、血压、血氧等生命体征数据。"
        "系统会基于这些指标，自动计算患者的初步急诊分级，一般划分为1至4级。"
        "对于被评定为3级或4级的普通急诊患者，系统会引导其前往自助挂号机或者人工窗口完成挂号操作。"
        "在挂号过程中，患者可以选择微信、支付宝以及医保卡等方式进行支付。"
        "支付完成后，系统会生成对应的急诊挂号流水号，并且将患者信息推送至相关科室的医生工作站排队系统。"
        "医生在工作站接诊患者时，可以查看其分诊记录，并根据实际情况开具电子处方。"
        "通过这一流程，患者的就诊信息可以在系统中实现连续记录与统一管理。"
        "对于被评定为1级或2级的危重症患者，处理流程会有所不同。"
        "此时护士需要立即安排抢救，系统应当支持跳过缴费环节，直接为患者建立临时急诊档案。"
        "等抢救结束之后，系统还需要支持补缴挂号费用以及抢救费用，以保证业务流程的完整性。"
        "在数据管理方面，所有进入急诊系统的患者档案与处方数据，都需要在每天零点进行自动同步，并上传至医院核心数据仓库进行统一备份。"
        "如果在同步过程中出现网络超时或者连接失败的情况，系统需要记录相关错误日志，同时通过短信方式通知运维人员进行处理。"
    )

    state = run_modeller_pipeline(
        requirement,
        config=PipelineConfig(provider="offline", max_iterations=2),
    )

    assert len(state["Requirement_Items"]) >= 4
    assert len(state["Use_Cases"]) >= 4
    use_case_names = [item.get("name") for item in state["Use_Cases"]]
    assert "录入急诊分诊信息" in use_case_names
    assert "办理急诊挂号支付" in use_case_names
    assert "建立临时急诊档案" in use_case_names
    assert "处理同步异常告警" in use_case_names


def test_pipeline_generates_exception_report_when_max_iterations_reached():
    state = run_modeller_pipeline(
        "当患者预约挂号成功时，系统应发送确认短信给患者",
        config=PipelineConfig(provider="offline", max_iterations=1, review_pass_threshold=101),
    )

    assert state["Status"] == "HUMAN_REVIEW_REQUIRED"
    assert state["Review_Exception_Report"]["type"] == "human_intervention_required"
    assert state["Review_Exception_Report"]["iteration_count"] == 1


def test_pipeline_can_disable_rag(monkeypatch):
    from rag.knowledge_base import knowledge_base

    def _unexpected_query(*args, **kwargs):
        raise AssertionError("knowledge_base.query should not be called when enable_rag=False")

    monkeypatch.setattr(knowledge_base, "query", _unexpected_query)

    state = run_modeller_pipeline(
        "当患者预约挂号成功时，系统应发送确认短信给患者",
        config=PipelineConfig(provider="offline", enable_rag=False, max_iterations=2),
    )

    assert state["RAG_Enabled"] is False
    assert state["Knowledge_Context"] == []


def test_review_step_allows_pass_with_only_warning_issues():
    class WarningOnlyReviewer:
        def review_model_pack(self, *args, **kwargs):
            return {
                "scores": {"overall": 82},
                "issues": ["第 3 行关系定义缺少说明标签或语义标注。"],
            }

    state = _review_step(
        {"UML_Artifacts": {"class_diagram": "@startuml\nclass A\n@enduml"}, "Requirement_Items": [], "EARS_Requirement": ""},
        WarningOnlyReviewer(),
        PipelineConfig(provider="offline", review_pass_threshold=75),
    )

    assert state["Status"] == "REVIEW_PASSED"
    assert state["Review_Blocking_Issues"] == []
    assert state["Review_Warnings"] == ["第 3 行关系定义缺少说明标签或语义标注。"]


def test_review_step_treats_coverage_issues_as_warnings():
    class CoverageWarningReviewer:
        def review_model_pack(self, *args, **kwargs):
            return {
                "scores": {"overall": 88},
                "issues": ["类图缺少核心类 Appointment，与当前 EARS 需求不一致。"],
            }

    state = _review_step(
        {"UML_Artifacts": {"class_diagram": "@startuml\nclass A\n@enduml"}, "Requirement_Items": [], "EARS_Requirement": ""},
        CoverageWarningReviewer(),
        PipelineConfig(provider="offline", review_pass_threshold=70),
    )

    assert state["Status"] == "REVIEW_PASSED"
    assert state["Review_Blocking_Issues"] == []
    assert state["Review_Warnings"] == ["类图缺少核心类 Appointment，与当前 EARS 需求不一致。"]


def test_review_step_blocks_on_real_structure_issues():
    class BlockingReviewer:
        def review_model_pack(self, *args, **kwargs):
            return {
                "scores": {"overall": 88},
                "issues": ["缺少 @startuml 或 @enduml 包裹。"],
            }

    state = _review_step(
        {"UML_Artifacts": {"class_diagram": "class A"}, "Requirement_Items": [], "EARS_Requirement": ""},
        BlockingReviewer(),
        PipelineConfig(provider="offline", review_pass_threshold=70),
    )

    assert state["Status"] == "NEEDS_REVISION"
    assert state["Review_Blocking_Issues"] == ["缺少 @startuml 或 @enduml 包裹。"]


def test_review_step_routes_requirement_issues_to_analyst():
    class RequirementIssueReviewer:
        def review_model_pack(self, *args, **kwargs):
            return {
                "scores": {"overall": 88},
                "issues": ["需求信息不完整，缺少异常路径。"],
            }

    state = _review_step(
        {"UML_Artifacts": {}, "Requirement_Items": [], "EARS_Requirement": ""},
        RequirementIssueReviewer(),
        PipelineConfig(provider="offline", review_pass_threshold=70),
    )

    assert state["Status"] == "NEEDS_REQUIREMENT_REVISION"
    assert state["Review_Target"] == "analyst"
    assert state["Review_Blocking_Issues"] == ["需求信息不完整，缺少异常路径。"]


def test_domain_knowledge_alignment_rewards_domain_entity_coverage(monkeypatch):
    from rag.knowledge_base import knowledge_base

    def _fake_query(*args, **kwargs):
        return [
            {
                "id": "KB-NOTIFY",
                "content": "挂号成功后应发送通知，建模时应体现 Notification 实体与通知机制。",
                "metadata": {"tags": ["通知", "短信", "挂号"], "type": "标准"},
            },
            {
                "id": "KB-PAY",
                "content": "支付场景应体现 BillingRecord 与结算状态。",
                "metadata": {"tags": ["支付", "结算"], "type": "标准"},
            },
        ]

    monkeypatch.setattr(knowledge_base, "query", _fake_query)

    reviewer = ReviewAgent(provider="offline")
    requirement_items = [
        RequirementUtils.analyze_requirement("当患者预约挂号成功后，系统应发送确认短信并更新支付状态", index=1),
    ]
    weak_pack = {
        "use_case_diagram": "@startuml\nactor Patient\nusecase UC1\nPatient --> UC1\n@enduml",
        "class_diagram": "@startuml\nclass Patient {\n  +patientId: String\n}\n@enduml",
        "sequence_diagram": "@startuml\nactor Patient\nparticipant System\nPatient -> System: 请求\n@enduml",
    }
    rich_pack = {
        "use_case_diagram": "@startuml\nactor Patient\nusecase UC1\nPatient --> UC1\n@enduml",
        "class_diagram": "@startuml\nclass Patient {\n  +patientId: String\n}\nclass Appointment {\n  +createAppointment()\n}\nclass Notification {\n  +sendNotification()\n}\nclass BillingRecord {\n  +updatePaymentStatus()\n}\nPatient \"1\" -- \"*\" Appointment : creates\nAppointment \"1\" --> \"*\" Notification : triggers\nPatient \"1\" -- \"*\" BillingRecord : pays\n@enduml",
        "sequence_diagram": "@startuml\nactor Patient\nparticipant System\nPatient -> System: 挂号并支付\nSystem --> Patient: 发送通知\n@enduml",
    }

    weak_score = reviewer.score_domain_knowledge_alignment(weak_pack, requirement_items=requirement_items)["score"]
    rich_score = reviewer.score_domain_knowledge_alignment(rich_pack, requirement_items=requirement_items)["score"]

    assert rich_score > weak_score
    assert rich_score - weak_score >= 20


def test_review_model_pack_overall_includes_domain_knowledge_alignment(monkeypatch):
    from rag.knowledge_base import knowledge_base

    def _fake_query(*args, **kwargs):
        return [
            {
                "id": "KB-NOTIFY",
                "content": "挂号成功后应发送通知，建模时应体现 Notification 实体与通知机制。",
                "metadata": {"tags": ["通知", "短信", "挂号"], "type": "标准"},
            }
        ]

    monkeypatch.setattr(knowledge_base, "query", _fake_query)

    reviewer = ReviewAgent(provider="offline")
    requirement_items = [
        RequirementUtils.analyze_requirement("当患者预约挂号成功后，系统应发送确认短信", index=1),
    ]
    uml_pack = {
        "use_case_diagram": "@startuml\nactor Patient\nusecase UC1\nPatient --> UC1 : 发起挂号\n@enduml",
        "class_diagram": "@startuml\nclass Patient {\n  +patientId: String\n}\nclass Appointment {\n  +createAppointment()\n}\nclass Notification {\n  +sendNotification()\n}\nPatient \"1\" -- \"*\" Appointment : creates\nAppointment --> Notification : triggers\n@enduml",
        "sequence_diagram": "@startuml\nactor Patient\nparticipant System\nPatient -> System: 提交挂号\nSystem --> Patient: 发送通知\n@enduml",
    }

    review = reviewer.review_model_pack(
        uml_pack,
        requirement_items=requirement_items,
        ears_requirement=requirement_items[0]["ears_requirement"],
    )

    scores = review["scores"]
    expected_overall = round(
        scores["diagram_consistency"] * 0.7
        + scores["domain_knowledge_alignment"] * 0.3
        + scores.get("coordination_bonus", 0)
    )

    assert scores["overall"] == expected_overall
