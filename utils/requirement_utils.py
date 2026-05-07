# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: utils/requirement_utils.py
Author: SeaOcean
Create Date: 2026-03-07
Description：需求解析与 EARS 转换工具
-------------------------------------------------
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from config import Config


class RequirementUtils:
    """需求处理与论文制品生成工具。"""

    EXPLICIT_SYSTEM_MARKERS = (
        "系统应当",
        "系统必须",
        "系统应",
        "系统需",
        "系统需要",
        "系统会",
    )

    DETAIL_MARKERS = (
        "必须",
        "不得",
        "仅",
        "至少",
        "不超过",
        "秒",
        "分钟",
        "小时",
        "失败",
        "异常",
        "超时",
        "告警",
        "日志",
        "通知",
        "回退",
        "补缴",
        "同步",
    )

    VAGUE_PHRASES = (
        "执行操作",
        "进行处理",
        "业务处理",
        "相应处理",
        "相关信息",
        "相关数据",
        "相关内容",
        "功能模块",
        "业务请求",
    )

    REQUIREMENT_BLOCK_START_PATTERNS = (
        r"^当",
        r"^如果",
        r"^若",
        r"^对于",
        r"^在数据管理方面",
        r"^在.+方面",
        r"^每日",
        r"^每当",
    )

    USE_CASE_NAME_PATTERNS = [
        (("分诊", "生命体征"), "录入急诊分诊信息"),
        (("分诊", "急诊分级"), "执行急诊分级"),
        (("挂号", "支付"), "办理急诊挂号支付"),
        (("挂号", "流水号"), "生成急诊挂号流水"),
        (("抢救", "临时急诊档案"), "建立临时急诊档案"),
        (("抢救", "补缴"), "补缴急诊抢救费用"),
        (("数据仓库", "同步"), "同步急诊业务数据"),
        (("错误日志", "短信"), "处理同步异常告警"),
        (("同步", "运维"), "处理同步异常告警"),
        (("预约", "通知"), "发送预约确认通知"),
        (("挂号", "通知"), "发送挂号业务通知"),
        (("病历", "查看"), "查看电子病历"),
        (("病历", "更新"), "更新电子病历"),
        (("支付", "失败"), "处理支付异常"),
        (("支付", "结算"), "执行支付结算"),
        (("登录", "身份"), "执行身份认证"),
    ]

    ACTOR_KEYWORDS = [
        "患者",
        "医生",
        "护士",
        "药师",
        "管理员",
        "前台",
        "系统",
        "家属",
    ]

    DOMAIN_KEYWORDS = {
        "登录认证": ["登录", "认证", "身份验证", "鉴权", "账号"],
        "预约挂号": ["预约", "挂号", "号源", "就诊"],
        "分诊管理": ["分诊", "排队", "叫号", "优先级"],
        "电子病历": ["病历", "健康记录", "电子病历", "诊断记录"],
        "处方管理": ["处方", "药品", "配药", "发药"],
        "通知提醒": ["短信", "通知", "提醒", "消息"],
        "计费支付": ["账单", "费用", "支付", "医保", "结算"],
    }

    ENTITY_LIBRARY = {
        "Patient": {
            "display_name": "Patient",
            "attributes": ["+patientId: String", "+name: String", "+phone: String"],
        },
        "Doctor": {
            "display_name": "Doctor",
            "attributes": ["+doctorId: String", "+name: String", "+department: String"],
        },
        "Nurse": {
            "display_name": "Nurse",
            "attributes": ["+nurseId: String", "+name: String", "+station: String"],
        },
        "Appointment": {
            "display_name": "Appointment",
            "attributes": ["+appointmentId: String", "+timeSlot: String", "+status: String"],
        },
        "MedicalRecord": {
            "display_name": "MedicalRecord",
            "attributes": ["+recordId: String", "+summary: String", "+updatedAt: DateTime"],
        },
        "TriageTask": {
            "display_name": "TriageTask",
            "attributes": ["+taskId: String", "+priority: String", "+queueNumber: String"],
        },
        "Prescription": {
            "display_name": "Prescription",
            "attributes": ["+prescriptionId: String", "+dosage: String", "+status: String"],
        },
        "Notification": {
            "display_name": "Notification",
            "attributes": ["+notificationId: String", "+channel: String", "+status: String"],
        },
        "AuthSession": {
            "display_name": "AuthSession",
            "attributes": ["+sessionId: String", "+token: String", "+expiredAt: DateTime"],
        },
        "BillingRecord": {
            "display_name": "BillingRecord",
            "attributes": ["+billingId: String", "+amount: Decimal", "+paymentStatus: String"],
        },
    }

    RELATIONSHIP_LIBRARY = {
        ("Patient", "Appointment"): 'Patient "1" -- "*" Appointment : creates',
        ("Doctor", "Appointment"): 'Doctor "1" -- "*" Appointment : handles',
        ("Patient", "MedicalRecord"): 'Patient "1" -- "*" MedicalRecord : owns',
        ("Doctor", "MedicalRecord"): 'Doctor "1" -- "*" MedicalRecord : updates',
        ("Patient", "Prescription"): 'Patient "1" -- "*" Prescription : receives',
        ("Doctor", "Prescription"): 'Doctor "1" -- "*" Prescription : writes',
        ("Patient", "Notification"): 'Patient "1" -- "*" Notification : receives',
        ("Patient", "AuthSession"): 'Patient "1" -- "*" AuthSession : starts',
        ("Patient", "BillingRecord"): 'Patient "1" -- "*" BillingRecord : pays',
        ("Nurse", "TriageTask"): 'Nurse "1" -- "*" TriageTask : executes',
        ("Patient", "TriageTask"): 'Patient "1" -- "*" TriageTask : enters',
        ("Appointment", "Notification"): 'Appointment "1" --> "*" Notification : triggers',
        ("BillingRecord", "Notification"): 'BillingRecord "1" --> "*" Notification : triggers',
        ("TriageTask", "Notification"): 'TriageTask "1" --> "*" Notification : triggers',
        ("Prescription", "MedicalRecord"): 'Prescription "*" --> "1" MedicalRecord : references',
    }

    ENTITY_KEYWORD_LIBRARY = {
        "AuthSession": ["登录", "认证", "身份验证", "鉴权", "令牌", "token"],
        "Appointment": ["预约", "挂号", "号源", "就诊"],
        "MedicalRecord": ["病历", "健康记录", "诊断记录", "电子病历"],
        "TriageTask": ["分诊", "叫号", "排队", "优先级", "流水号"],
        "Prescription": ["处方", "药品", "配药", "发药"],
        "Notification": ["短信", "通知", "提醒", "消息", "推送"],
        "BillingRecord": ["支付", "费用", "账单", "结算", "医保"],
    }

    ENTITY_METHOD_LIBRARY = {
        "AuthSession": [
            (("登录", "认证", "身份验证", "鉴权"), "+verifyIdentity()"),
            (("令牌", "token"), "+issueToken()"),
        ],
        "Appointment": [
            (("预约", "挂号"), "+createAppointment()"),
            (("号源", "锁定"), "+lockSlot()"),
            (("号源", "释放"), "+releaseSlot()"),
            (("取消",), "+cancelAppointment()"),
        ],
        "MedicalRecord": [
            (("病历", "查看"), "+queryRecord()"),
            (("病历", "更新"), "+updateRecord()"),
            (("病历", "写入"), "+writeRecord()"),
        ],
        "TriageTask": [
            (("分诊",), "+createTriageTask()"),
            (("优先级",), "+updatePriority()"),
            (("流水号", "叫号", "排队"), "+generateQueueNumber()"),
        ],
        "Prescription": [
            (("处方", "开具"), "+issuePrescription()"),
            (("药品", "校验"), "+validateMedication()"),
            (("发药",), "+dispenseMedication()"),
        ],
        "Notification": [
            (("短信", "通知", "提醒", "消息", "推送"), "+sendNotification()"),
        ],
        "BillingRecord": [
            (("支付", "结算"), "+processPayment()"),
            (("支付状态", "状态"), "+updatePaymentStatus()"),
            (("回滚", "退款", "取消"), "+rollbackPayment()"),
        ],
    }

    @staticmethod
    def split_requirements(requirement_text: str) -> list[str]:
        text = (requirement_text or "").replace("\r", "\n").strip()
        if not text:
            return []

        chunks = re.split(r"[\n；;]+", text)
        requirements: list[str] = []
        for chunk in chunks:
            cleaned = re.sub(r"^\s*(\d+[\.\、\)]|[-*])\s*", "", chunk.strip())
            cleaned = cleaned.strip("。 ")
            if cleaned:
                requirements.extend(RequirementUtils._split_complex_requirement_chunk(cleaned))
        return requirements

    @staticmethod
    def _split_complex_requirement_chunk(chunk: str) -> list[str]:
        if not chunk:
            return []

        sentences = [
            sentence.strip().strip("。 ")
            for sentence in re.split(r"(?<=[。！？!?])\s*", chunk)
            if sentence and sentence.strip().strip("。 ")
        ]
        if len(sentences) <= 1:
            return [chunk.strip().strip("。 ")]

        requirements: list[str] = []
        current_sentences: list[str] = []
        start_pattern = re.compile("|".join(RequirementUtils.REQUIREMENT_BLOCK_START_PATTERNS))

        for sentence in sentences:
            is_new_block = bool(current_sentences) and bool(start_pattern.match(sentence))
            if is_new_block:
                requirements.append("。".join(current_sentences).strip("。 "))
                current_sentences = [sentence]
                continue
            current_sentences.append(sentence)

        if current_sentences:
            requirements.append("。".join(current_sentences).strip("。 "))

        return [item for item in requirements if item]

    @staticmethod
    def _clean_phrase(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip("，,。；; "))
        return cleaned

    @staticmethod
    def _extract_condition(requirement_text: str) -> str:
        match = re.search(
            r"(?:当|如果|若|在)(.+?)(?:时|后|成功时|成功后|失败时|失败后|触发时|提交后|完成后|，|,)",
            requirement_text,
        )
        if match:
            return RequirementUtils._clean_phrase(match.group(1))
        return "系统接收到业务请求"

    @staticmethod
    def _extract_action_segment(requirement_text: str) -> str:
        markers = ["系统应当", "系统必须", "系统应", "系统需", "系统需要", "应当", "必须", "应", "需"]
        for marker in markers:
            if marker in requirement_text:
                return RequirementUtils._clean_phrase(requirement_text.split(marker, 1)[1])
        return RequirementUtils._clean_phrase(requirement_text)

    @staticmethod
    def _extract_actions(requirement_text: str) -> list[str]:
        segment = RequirementUtils._extract_action_segment(requirement_text)
        candidates = re.split(r"(?:并且|并|且|同时|以及|然后|随后)", segment)
        actions = [RequirementUtils._clean_phrase(item) for item in candidates if RequirementUtils._clean_phrase(item)]
        if actions:
            return actions

        keyword_actions = [
            ("登录", "验证用户身份"),
            ("验证", "校验用户身份"),
            ("显示", "返回业务结果"),
            ("预约", "创建预约记录"),
            ("挂号", "创建挂号记录"),
            ("分诊", "生成分诊任务"),
            ("病历", "更新病历信息"),
            ("短信", "发送短信通知"),
            ("通知", "发送业务通知"),
        ]
        inferred = [value for keyword, value in keyword_actions if keyword in requirement_text]
        return inferred or ["执行对应业务操作"]

    @staticmethod
    def _extract_actors(requirement_text: str) -> list[str]:
        actors = [actor for actor in RequirementUtils.ACTOR_KEYWORDS if actor in requirement_text]
        if "系统" not in actors:
            actors.append("系统")
        return actors

    @staticmethod
    def parse_natural_language(requirement_text: str) -> dict[str, Any]:
        requirement_text = RequirementUtils._clean_phrase(requirement_text)
        actions = RequirementUtils._extract_actions(requirement_text)
        condition = RequirementUtils._extract_condition(requirement_text)
        actors = RequirementUtils._extract_actors(requirement_text)
        constraints = []
        for keyword in ("必须", "不得", "仅", "至少", "不超过", "秒", "分钟", "小时"):
            if keyword in requirement_text:
                constraints.append(keyword)

        primary_actor = next((actor for actor in actors if actor != "系统"), "用户")
        return {
            "original": requirement_text,
            "components": {
                "actors": actors,
                "actions": actions,
                "conditions": [condition],
                "constraints": constraints,
            },
            "primary_actor": primary_actor,
            "domain": RequirementUtils.extract_domain(requirement_text),
        }

    @staticmethod
    def convert_to_ears(requirement_text: str) -> str:
        parsed = RequirementUtils.parse_natural_language(requirement_text)
        condition = parsed["components"]["conditions"][0]
        actions = parsed["components"]["actions"] or ["执行对应业务操作"]
        action_line = f"系统应{actions[0]}"
        additional_actions = "\n".join(f"AND 系统应{action}" for action in actions[1:])
        return Config.get_ears_template().format(
            condition=condition,
            action=action_line,
            additional_actions=additional_actions,
        ).strip()

    @staticmethod
    def build_acceptance_criteria(analysis: dict[str, Any]) -> list[str]:
        actor = analysis.get("primary_actor", "用户")
        condition = analysis.get("components", {}).get("conditions", ["系统接收到业务请求"])[0]
        actions = analysis.get("components", {}).get("actions", [])
        criteria = [f"Given {condition}, when {actor}发起业务操作, then 系统应稳定响应。"]
        for action in actions:
            criteria.append(f"系统需完成“{action}”并输出可验证结果。")
        return criteria

    @staticmethod
    def build_use_case(analysis: dict[str, Any], index: int = 1) -> dict[str, Any]:
        actions = analysis.get("components", {}).get("actions", ["执行业务流程"])
        condition = analysis.get("components", {}).get("conditions", ["系统接收到业务请求"])[0]
        supporting_actors = [
            actor
            for actor in analysis.get("components", {}).get("actors", [])
            if actor not in {analysis.get("primary_actor"), "系统"}
        ]
        name = RequirementUtils.normalize_use_case_name(
            actions[0],
            original_text=analysis.get("original", ""),
            domain=analysis.get("domain", ""),
            actions=actions,
        )
        return {
            "id": f"UC{index:03d}",
            "name": name,
            "goal": f"{analysis.get('primary_actor', '用户')}完成{name}",
            "scope": "Med-MARAG 医疗需求建模平台",
            "level": "Primary task",
            "primary_actor": analysis.get("primary_actor", "用户"),
            "supporting_actors": supporting_actors,
            "preconditions": [condition],
            "main_flow": [
                f"{analysis.get('primary_actor', '用户')}触发业务请求。",
                f"系统根据条件“{condition}”校验上下文。",
                *[f"系统执行：{action}。" for action in actions],
            ],
            "postconditions": [f"系统完成{name}并留下可追溯记录。"],
        }

    @staticmethod
    def normalize_use_case_name(
        action_text: str,
        *,
        original_text: str = "",
        domain: str = "",
        actions: list[str] | None = None,
    ) -> str:
        source_text = " ".join(part for part in [original_text, domain, *(actions or [])] if part).strip()
        for keywords, canonical_name in RequirementUtils.USE_CASE_NAME_PATTERNS:
            if all(keyword in source_text for keyword in keywords):
                return canonical_name

        text = action_text.replace("系统应", "").replace("系统", "").strip()
        replacements = [
            ("发送确认短信", "发送确认通知"),
            ("发短信", "发送短信通知"),
            ("给患者", ""),
            ("给医生", ""),
            ("给用户", ""),
            ("短信", "短信通知"),
            ("显示", "展示"),
            ("看", "查看"),
            ("改", "更新"),
            ("弄", "处理"),
            ("在系统中", ""),
            ("需要", ""),
            ("要", ""),
            ("对应的", ""),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        text = text.replace("通知通知", "通知")
        text = re.sub(r"^(支持|实现|完成|进行)", "", text)
        text = re.sub(r"(。.*)$", "", text)
        text = re.sub(r"(，.*)$", "", text)
        text = re.sub(r"\s+", "", text)
        text = text.strip("，,。；;")
        text = RequirementUtils._normalize_use_case_name_style(text)
        if len(text) > 18:
            text = text[:18]
        return text or "执行业务操作"

    @staticmethod
    def _normalize_use_case_name_style(text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return ""

        verb_prefixes = (
            "录入",
            "执行",
            "办理",
            "生成",
            "建立",
            "补缴",
            "同步",
            "处理",
            "发送",
            "查看",
            "更新",
            "维护",
            "创建",
            "查询",
            "通知",
            "校验",
            "提交",
        )
        if normalized.startswith(verb_prefixes):
            return normalized

        if any(keyword in normalized for keyword in ("短信", "消息", "提醒", "通知")):
            return f"发送{normalized}"
        if "病历" in normalized:
            return f"处理{normalized}"
        if "挂号" in normalized or "预约" in normalized:
            return f"办理{normalized}"
        if "支付" in normalized or "费用" in normalized or "结算" in normalized:
            return f"处理{normalized}"
        if "日志" in normalized or "告警" in normalized:
            return f"处理{normalized}"
        if "同步" in normalized:
            return f"执行{normalized}"
        if "档案" in normalized or "记录" in normalized:
            return f"建立{normalized}"
        return f"处理{normalized}"

    @staticmethod
    def analyze_requirement(requirement_text: str, index: int = 1) -> dict[str, Any]:
        parsed = RequirementUtils.parse_natural_language(requirement_text)
        requirement_id = RequirementUtils.generate_requirement_id(index=index)
        ears_requirement = RequirementUtils.convert_to_ears(requirement_text)
        validation = RequirementUtils.validate_requirement(requirement_text)
        use_case = RequirementUtils.build_use_case(parsed, index=index)
        analysis = {
            "id": requirement_id,
            "original": parsed["original"],
            "domain": parsed["domain"],
            "primary_actor": parsed["primary_actor"],
            "actors": parsed["components"]["actors"],
            "actions": parsed["components"]["actions"],
            "condition": parsed["components"]["conditions"][0],
            "constraints": parsed["components"]["constraints"],
            "ears_requirement": ears_requirement,
            "validation": validation,
            "use_case": use_case,
            "acceptance_criteria": RequirementUtils.build_acceptance_criteria(parsed),
        }
        return analysis

    @staticmethod
    def validate_requirement(requirement_text: str) -> dict[str, Any]:
        raw_text = RequirementUtils._clean_phrase(requirement_text or "")
        issues: list[str] = []
        parsed = RequirementUtils.parse_natural_language(raw_text)

        explicit_condition = RequirementUtils._has_explicit_condition(raw_text)
        explicit_system = any(marker in raw_text for marker in RequirementUtils.EXPLICIT_SYSTEM_MARKERS)
        actions = parsed["components"]["actions"]
        non_generic_actions = [action for action in actions if action and action != "执行对应业务操作"]
        entity_hits = RequirementUtils._count_specific_entities(raw_text)
        detail_hits = sum(1 for marker in RequirementUtils.DETAIL_MARKERS if marker in raw_text)
        vague_hits = sum(1 for phrase in RequirementUtils.VAGUE_PHRASES if phrase in raw_text)
        connector_hits = len(re.findall(r"(?:并且|并|且|同时|以及|然后|随后)", raw_text))

        if len(raw_text) < 10:
            issues.append("需求描述过短，建议补充上下文。")
        if not explicit_condition:
            issues.append("缺少显式触发条件，建议使用“当/如果/若”等条件表达。")
        if not explicit_system:
            issues.append("缺少显式系统主体，建议使用“系统应/系统必须”等规范表达。")
        if not non_generic_actions:
            issues.append("未识别到明确动作，建议补充系统行为。")
        if entity_hits == 0:
            issues.append("业务对象不够明确，建议补充患者、病历、支付、通知等核心对象。")
        if vague_hits > 0:
            issues.append("存在较笼统表述，建议替换为可验证的具体业务动作。")

        score = 36
        if len(raw_text) >= 10:
            score += 4
        if explicit_condition:
            score += 10
        if explicit_system:
            score += 10
        if non_generic_actions:
            score += min(10, 4 + len(non_generic_actions) * 3)
        if entity_hits > 0:
            score += min(8, entity_hits * 2)
        if detail_hits > 0:
            score += min(8, detail_hits)
        if connector_hits > 0:
            score += min(3, connector_hits)

        score -= vague_hits * 8
        if len(raw_text) > 80 and detail_hits == 0:
            score -= 6
        if len(raw_text) > 140 and connector_hits >= 2 and detail_hits <= 1:
            score -= 6

        score = max(35, min(92, score))
        return {
            "valid": len(issues) == 0 and score >= 70,
            "issues": issues,
            "score": score,
        }

    @staticmethod
    def _has_explicit_condition(text: str) -> bool:
        if not text:
            return False
        patterns = (
            r"(^|[，,。；;\s])当.+?(时|后)",
            r"(^|[，,。；;\s])如果.+",
            r"(^|[，,。；;\s])若.+",
            r"(^|[，,。；;\s])一旦.+",
            r"(^|[，,。；;\s])对于.+",
            r"(^|[，,。；;\s])在.+?(时|过程中|方面)",
            r".+?(完成后|成功后|失败后|失败时|触发时)",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _count_specific_entities(text: str) -> int:
        entity_keywords: set[str] = set()
        for actor in ("患者", "医生", "护士", "药师", "管理员", "家属"):
            if actor in text:
                entity_keywords.add(actor)
        for keywords in RequirementUtils.ENTITY_KEYWORD_LIBRARY.values():
            for keyword in keywords:
                if keyword in text:
                    entity_keywords.add(keyword)
        return len(entity_keywords)

    @staticmethod
    def generate_requirement_id(prefix: str = "REQ", index: int | None = None, requirement_text: str = "") -> str:
        if index is not None:
            return f"{prefix}{index:03d}"
        digest = hashlib.md5(requirement_text.encode("utf-8")).hexdigest()[:6].upper()
        return f"{prefix}{digest}"

    @staticmethod
    def extract_domain(requirement_text: str) -> str:
        for domain, keywords in RequirementUtils.DOMAIN_KEYWORDS.items():
            if any(keyword in requirement_text for keyword in keywords):
                return domain
        return "通用医疗流程"

    @staticmethod
    def _detect_entity_names(text: str, actors: list[str]) -> set[str]:
        entity_names: set[str] = set()
        if "患者" in actors:
            entity_names.add("Patient")
        if "医生" in actors:
            entity_names.add("Doctor")
        if "护士" in actors:
            entity_names.add("Nurse")

        for entity_name, keywords in RequirementUtils.ENTITY_KEYWORD_LIBRARY.items():
            if any(keyword in text for keyword in keywords):
                entity_names.add(entity_name)
        return entity_names

    @staticmethod
    def _infer_methods(entity_name: str, text: str, actions: list[str]) -> list[str]:
        merged_text = f"{text} {' '.join(actions)}"
        methods: list[str] = []
        for keywords, signature in RequirementUtils.ENTITY_METHOD_LIBRARY.get(entity_name, []):
            if any(keyword in merged_text for keyword in keywords):
                methods.append(signature)
        return list(dict.fromkeys(methods))

    @staticmethod
    def infer_entities(requirement_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entity_specs: dict[str, dict[str, Any]] = {}
        for item in requirement_items:
            actors = item.get("actors", [])
            text = f"{item.get('original', '')} {' '.join(item.get('actions', []))}"
            entity_names = RequirementUtils._detect_entity_names(text, actors)
            for name in entity_names:
                definition = RequirementUtils.ENTITY_LIBRARY.get(name, {"attributes": []})
                spec = entity_specs.setdefault(
                    name,
                    {
                        "name": name,
                        "attributes": list(definition.get("attributes", [])),
                        "methods": [],
                        "relationships": [],
                    },
                )
                spec["methods"] = list(
                    dict.fromkeys(spec["methods"] + RequirementUtils._infer_methods(name, text, item.get("actions", [])))
                )

        if not entity_specs:
            for fallback_name in ("Patient", "Appointment", "Notification"):
                definition = RequirementUtils.ENTITY_LIBRARY.get(fallback_name, {"attributes": []})
                entity_specs[fallback_name] = {
                    "name": fallback_name,
                    "attributes": list(definition.get("attributes", [])),
                    "methods": [],
                    "relationships": [],
                }

        return [entity_specs[name] for name in sorted(entity_specs)]

    @staticmethod
    def infer_relationships(
        entity_names: list[str],
        requirement_items: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        relations: list[str] = []
        pairs = set(RequirementUtils.RELATIONSHIP_LIBRARY)
        for source, target in pairs:
            if source in entity_names and target in entity_names:
                relations.append(RequirementUtils.RELATIONSHIP_LIBRARY[(source, target)])

        if requirement_items:
            merged_text = " ".join(
                f"{item.get('original', '')} {' '.join(item.get('actions', []))}" for item in requirement_items
            )
            dynamic_relations = [
                (("医生", "病历"), 'Doctor "1" --> "*" MedicalRecord : reviews'),
                (("医生", "预约"), 'Doctor "1" --> "*" Appointment : schedules'),
                (("系统", "通知"), 'Appointment "1" --> "*" Notification : notifies'),
                (("支付", "通知"), 'BillingRecord "1" --> "*" Notification : notifies'),
                (("分诊", "通知"), 'TriageTask "1" --> "*" Notification : notifies'),
            ]
            for keywords, relation in dynamic_relations:
                if all(keyword in merged_text for keyword in keywords):
                    relations.append(relation)

        return list(dict.fromkeys(relations))
