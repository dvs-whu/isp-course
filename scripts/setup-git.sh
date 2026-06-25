#!/bin/bash
# 智能信号处理课程Git初始化脚本
# 用法: ./scripts/setup-git.sh [github_repo_url]

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# 切换到vault目录
VAULT_DIR="/Users/leiyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/智能信号处理"
cd "$VAULT_DIR"

echo -e "${BLUE}🔧 智能信号处理课程Git初始化工具${NC}"
echo "======================================"
echo ""

# 检查是否已初始化
if [[ ! -d ".git" ]]; then
    echo -e "${BLUE}📦 初始化Git仓库...${NC}"
    git init
    git branch -M main
fi

# 触发iCloud文件下载
echo -e "${BLUE}☁️  触发iCloud文件下载...${NC}"
echo "   这可能需要几分钟..."
find . -name "*.md" -exec brctl download {} \;
find . -name "*.py" -exec brctl download {} \;
find . -name "*.png" -exec brctl download {} \;
find . -name "*.svg" -exec brctl download {} \;
find . -name "*.sh" -exec brctl download {} \;

echo ""
echo -e "${YELLOW}⏳ 等待iCloud同步...${NC}"
sleep 5

# 添加所有文件
echo ""
echo -e "${BLUE}📁 添加文件到Git...${NC}"
git add -A

# 显示状态
echo ""
echo -e "${BLUE}📊 Git状态:${NC}"
git status

# 获取GitHub仓库URL
if [[ -n "$1" ]]; then
    GITHUB_URL="$1"
else
    echo ""
    read -p "🔗 请输入GitHub仓库URL (或按回车跳过): " GITHUB_URL
fi

# 设置远程仓库
if [[ -n "$GITHUB_URL" ]]; then
    echo ""
    echo -e "${BLUE}🔗 设置远程仓库...${NC}"
    git remote remove origin 2>/dev/null || true
    git remote add origin "$GITHUB_URL"
    echo -e "${GREEN}✅ 远程仓库已设置: $GITHUB_URL${NC}"
fi

# 提示用户提交
echo ""
echo -e "${GREEN}✅ Git初始化完成！${NC}"
echo ""
echo "📋 下一步操作:"
echo ""
echo "  1. 检查文件列表:"
echo "     git status"
echo ""
echo "  2. 首次提交:"
echo "     git commit -m 'feat: 初始化智能信号处理课程仓库'"
echo ""
echo "  3. 推送到GitHub:"
echo "     git push -u origin main"
echo ""
echo "  4. 创建版本标签:"
echo "     ./scripts/tag-version.sh v1.0.0 '2026秋季学期初始版本'"
echo ""
echo "🔗 GitHub仓库: ${GITHUB_URL:-未设置}"
