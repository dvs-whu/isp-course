#!/bin/bash
# 从Obsidian同步内容到MkDocs
# 用法: ./scripts/sync-to-mkdocs.sh

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

VAULT_DIR="/Users/leiyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/智能信号处理"
DOCS_DIR="$VAULT_DIR/docs"

echo -e "${BLUE}🔄 同步Obsidian内容到MkDocs${NC}"
echo "================================"
echo ""

# 同步知识点
echo -e "${BLUE}📚 同步知识点...${NC}"

# 经典信号处理
for file in "$VAULT_DIR/知识点/经典信号处理/"*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$DOCS_DIR/knowledge/classic/$filename"
        echo "  ✅ classic/$filename"
    fi
done

# 时频分析
for file in "$VAULT_DIR/知识点/时频分析/"*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$DOCS_DIR/knowledge/time-freq/$filename"
        echo "  ✅ time-freq/$filename"
    fi
done

# 统计信号处理
for file in "$VAULT_DIR/知识点/统计信号处理/"*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$DOCS_DIR/knowledge/statistical/$filename"
        echo "  ✅ statistical/$filename"
    fi
done

# 深度学习
for file in "$VAULT_DIR/知识点/深度学习/"*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$DOCS_DIR/knowledge/deep-learning/$filename"
        echo "  ✅ deep-learning/$filename"
    fi
done

# 图像信号处理
for file in "$VAULT_DIR/知识点/图像信号处理/"*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$DOCS_DIR/knowledge/image/$filename"
        echo "  ✅ image/$filename"
    fi
done

# 语音信号处理
for file in "$VAULT_DIR/知识点/语音信号处理/"*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$DOCS_DIR/knowledge/speech/$filename"
        echo "  ✅ speech/$filename"
    fi
done

# 事件驱动
for file in "$VAULT_DIR/知识点/事件驱动/"*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$DOCS_DIR/knowledge/event/$filename"
        echo "  ✅ event/$filename"
    fi
done

echo ""

# 同步项目
echo -e "${BLUE}📁 同步项目...${NC}"

# 项目一
if [ -f "$VAULT_DIR/课程/项目库/项目一/项目一_迷你项目选题.md" ]; then
    cp "$VAULT_DIR/课程/项目库/项目一/项目一_迷你项目选题.md" "$DOCS_DIR/projects/project1/overview.md"
    echo "  ✅ project1/overview.md"
fi

# 项目二
if [ -f "$VAULT_DIR/课程/项目库/项目二/项目二_跨学科项目选题指南.md" ]; then
    cp "$VAULT_DIR/课程/项目库/项目二/项目二_跨学科项目选题指南.md" "$DOCS_DIR/projects/project2/overview.md"
    echo "  ✅ project2/overview.md"
fi

echo ""

# 同步实验
echo -e "${BLUE}🧪 同步实验...${NC}"

if [ -f "$VAULT_DIR/课程/实验课程/实验大纲.md" ]; then
    cp "$VAULT_DIR/课程/实验课程/实验大纲.md" "$DOCS_DIR/labs/overview.md"
    echo "  ✅ labs/overview.md"
fi

echo ""

# 同步课程信息
echo -e "${BLUE}📋 同步课程信息...${NC}"

if [ -f "$VAULT_DIR/课程/理论课程/课程大纲_本科版.md" ]; then
    cp "$VAULT_DIR/课程/理论课程/课程大纲_本科版.md" "$DOCS_DIR/syllabus.md"
    echo "  ✅ syllabus.md"
fi

if [ -f "$VAULT_DIR/课程/13周教学进度表.md" ]; then
    cp "$VAULT_DIR/课程/13周教学进度表.md" "$DOCS_DIR/schedule.md"
    echo "  ✅ schedule.md"
fi

if [ -f "$VAULT_DIR/课程/考核评分标准.md" ]; then
    cp "$VAULT_DIR/课程/考核评分标准.md" "$DOCS_DIR/grading.md"
    echo "  ✅ grading.md"
fi

echo ""

# 同步资源
echo -e "${BLUE}📚 同步资源...${NC}"

if [ -f "$VAULT_DIR/知识点/Python信号处理速修.md" ]; then
    cp "$VAULT_DIR/知识点/Python信号处理速修.md" "$DOCS_DIR/resources/python.md"
    echo "  ✅ resources/python.md"
fi

if [ -f "$VAULT_DIR/知识点/概率论速修.md" ]; then
    cp "$VAULT_DIR/知识点/概率论速修.md" "$DOCS_DIR/resources/probability.md"
    echo "  ✅ resources/probability.md"
fi

if [ -f "$VAULT_DIR/知识点/复数与欧拉公式速修.md" ]; then
    cp "$VAULT_DIR/知识点/复数与欧拉公式速修.md" "$DOCS_DIR/resources/complex.md"
    echo "  ✅ resources/complex.md"
fi

echo ""
echo -e "${GREEN}✅ 同步完成！${NC}"
echo ""
echo "下一步："
echo "  1. 预览站点: mkdocs serve"
echo "  2. 构建站点: mkdocs build"
echo "  3. 部署: git add docs/ && git commit -m 'docs: 更新内容' && git push"
