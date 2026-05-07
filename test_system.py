#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: test_system.py
Author: SeaOcean
Create Date: 2026-03-07
Description：系统功能检查脚本
-------------------------------------------------
"""

import sys
import os
import traceback

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run_check(title: str, checker):
    """运行单项检查，兼容脚本模式与 pytest。"""
    print("=" * 50)
    print(f"测试{title}...")
    try:
        checker()
        print(f"✓ {title}测试通过")
        return True
    except Exception as e:
        print(f"✗ {title}测试失败: {e}")
        traceback.print_exc()
        return False

def _check_config():
    from config import Config

    provider = Config.get_requested_llm_provider()
    print(f"✓ Provider: {provider}")
    print(f"✓ API Key: {Config.get_llm_api_key(provider)[:20]}...")
    print(f"✓ Model: {Config.get_llm_model(provider) or 'N/A'}")
    print(f"✓ Base URL: {Config.get_llm_base_url(provider) or 'N/A'}")


def _check_utils():
    from utils.requirement_utils import RequirementUtils
    from utils.plantuml_utils import PlantUMLUtils

    test_req = "当患者登录系统时，系统应验证其身份并显示个人健康记录"
    parsed = RequirementUtils.parse_natural_language(test_req)
    print(f"✓ 需求解析: {parsed}")

    ears_req = RequirementUtils.convert_to_ears(test_req)
    print(f"✓ EARS 转换:\n{ears_req}")

    plantuml_code = PlantUMLUtils.generate_sequence_diagram(test_req)
    assert plantuml_code.startswith("@startuml")
    print("✓ PlantUML 生成成功")


def _check_rag():
    from rag.knowledge_base import knowledge_base

    query = "医疗软件需求标准"
    results = knowledge_base.query(query, n_results=3)
    assert len(results) > 0
    print(f"✓ 知识库查询结果: {len(results)} 条")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['content'][:50]}...")

    stats = knowledge_base.get_collection_stats()
    print(f"✓ 知识库统计: {stats}")


def _check_agents():
    from agents import analyst_agent, architect_agent, review_agent

    assert analyst_agent is not None
    assert architect_agent is not None
    assert review_agent is not None
    print("✓ 分析师智能体初始化成功")
    print("✓ 架构师智能体初始化成功")
    print("✓ 审查智能体初始化成功")


def _check_traceability():
    print("\n" + "=" * 50)
    print("测试追溯矩阵...")
    try:
        from traceability_matrix import traceability_matrix
        
        # 创建测试数据
        requirements = [
            {"id": "REQ001", "description": "患者登录功能"},
            {"id": "REQ002", "description": "健康记录显示"}
        ]
        
        design_elements = ["登录模块", "身份验证组件", "健康记录模块"]
        
        test_cases = [
            {"id": "TC001", "description": "验证登录功能"},
            {"id": "TC002", "description": "验证健康记录显示"}
        ]
        
        # 生成追溯矩阵
        matrix = traceability_matrix.generate_matrix(requirements, design_elements, test_cases)
        print(f"✓ 追溯矩阵生成: {len(matrix)} 条记录")
        
        # 获取 DataFrame
        df = traceability_matrix.get_matrix_dataframe()
        print(f"✓ DataFrame 形状: {df.shape}")
        print(df.head())
        
        # 验证追溯完整性
        validation = traceability_matrix.validate_traceability()
        print(f"✓ 追溯验证: {validation}")
        assert validation["valid"] is True
    except Exception:
        raise


def _check_test_case_generator():
    from test_case_generator import test_case_generator

    test_requirement = "IF 患者登录系统 THEN 系统应验证其身份 AND 系统应显示个人健康记录 ENDIF"
    test_cases = test_case_generator._generate_default_test_cases()
    assert len(test_cases) >= 2
    print(f"✓ 生成测试用例: {len(test_cases)} 个")
    for tc in test_cases:
        print(f"  - {tc['id']}: {tc['scenario']}")

    validation = test_case_generator.validate_test_cases(test_cases, test_requirement)
    print(f"✓ 测试用例验证: {validation}")
    assert validation["valid"] is True


def test_config():
    assert _run_check("配置文件", _check_config)


def test_utils():
    assert _run_check("工具函数", _check_utils)


def test_rag():
    assert _run_check("RAG 知识库", _check_rag)


def test_agents():
    assert _run_check("智能体", _check_agents)


def test_traceability():
    assert _run_check("追溯矩阵", _check_traceability)


def test_test_case_generator():
    assert _run_check("测试用例生成器", _check_test_case_generator)

def main():
    """主测试函数"""
    print("=" * 50)
    print("Med-MARAG 系统测试")
    print("=" * 50)
    
    results = []
    
    # 运行所有测试
    results.append(("配置文件", _run_check("配置文件", _check_config)))
    results.append(("工具函数", _run_check("工具函数", _check_utils)))
    results.append(("RAG 知识库", _run_check("RAG 知识库", _check_rag)))
    results.append(("智能体", _run_check("智能体", _check_agents)))
    results.append(("追溯矩阵", _run_check("追溯矩阵", _check_traceability)))
    results.append(("测试用例生成器", _run_check("测试用例生成器", _check_test_case_generator)))
    
    # 打印测试结果汇总
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"总计: {passed} 通过, {failed} 失败")
    print("=" * 50)
    
    if failed == 0:
        print("🎉 所有测试通过！系统运行正常。")
        return 0
    else:
        print(f"⚠️  {failed} 个测试失败，请检查相关模块。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
