#!/bin/bash
# 智能信号处理课程发布脚本
# 用法: ./scripts/publish.sh [commit_message]

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 切换到vault目录
VAULT_DIR="/Users/leiyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/智能信号处理"
cd "$VAULT_DIR"

echo -e "${BLUE}📚 智能信号处理课程发布工具${NC}"
echo "================================"
echo ""

# 检查git状态
if [[ -z $(git status -s) ]]; then
    echo -e "${GREEN}✅ 没有未提交的更改${NC}"
    exit 0
fi

# 显示更改
echo -e "${YELLOW}📋 检测到以下更改:${NC}"
git status -s
echo ""

# 获取提交信息
if [[ -n "$1" ]]; then
    COMMIT_MSG="$1"
else
    read -p "📝 请输入提交信息: " COMMIT_MSG
fi

if [[ -z "$COMMIT_MSG" ]]; then
    echo "❌ 提交信息不能为空"
    exit 1
fi

# 提交更改
echo ""
echo -e "${BLUE}📦 提交更改...${NC}"
git add -A
git commit -m "$COMMIT_MSG"

# 推送到GitHub
echo ""
echo -e "${BLUE}🚀 推送到GitHub...${NC}"
git push origin main

echo ""
echo -e "${GREEN}✅ 发布完成！${NC}"
echo ""
echo "📊 提交摘要:"
echo "   信息: $COMMIT_MSG"
echo "   时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "🔗 GitHub: https://github.com/YOUR_USERNAME/isp-course"
