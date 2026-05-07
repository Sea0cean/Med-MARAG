# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Med-MARAG
File Name: agents/__init__.py
Author: SeaOcean
Create Date: 2026-01-18
Description：Agent 包导出定义
-------------------------------------------------
"""
from .analyst_agent import AnalystAgent
from .architect_agent import ArchitectAgent
from .review_agent import ReviewAgent

# Avoid creating agent instances at import time.
# Agents require API keys; instantiate them explicitly when needed.
