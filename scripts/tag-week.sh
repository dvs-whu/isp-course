#!/bin/bash
# 智能信号处理课程周次标签脚本
# 用法: ./scripts/tag-week.sh [week_number]

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 切换到vault目录
VAULT_DIR="/Users/leiyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/智能信号处理"
cd "$VAULT_DIR"

echo -e "${BLUE}📅 智能信号处理课程周次标签工具${NC}"
echo "===================================="
echo ""

# 获取周次
if [[ -n "$1" ]]; then
    WEEK_NUM="$1"
else
    read -p "📅 请输入周次 (1-13): " WEEK_NUM
fi

# 验证周次
if [[ ! "$WEEK_NUM" =~ ^[0-9]+$ ]] || [[ "$WEEK_NUM" -lt 1 ]] || [[ "$WEEK_NUM" -gt 13 ]]; then
    echo "❌ 无效的周次，请输入1-13之间的数字"
    exit 1
fi

# 获取本周内容摘要
echo ""
echo "📝 请输入本周主要内容（按Enter结束）:"
read CONTENT

if [[ -z "$CONTENT" ]]; then
    echo "❌ 内容摘要不能为空"
    exit 1
fi

# 创建标签
TAG_NAME="week${WEEK_NUM}"
TAG_MSG="W${WEEK_NUM}完成: ${CONTENT}"

echo ""
echo -e "${BLUE}🏷️  创建标签...${NC}"
echo "   标签名: $TAG_NAME"
echo "   内容: $TAG_MSG"
echo ""

# 检查标签是否已存在
if git tag -l "$TAG_NAME" | grep -q "$TAG_NAME"; then
    echo -e "${YELLOW}⚠️  标签 $TAG_NAME 已存在${NC}"
    read -p "是否覆盖？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 操作取消"
        exit 1
    fi
    git tag -d "$TAG_NAME"
fi

# 创建新标签
git tag -a "$TAG_NAME" -m "$TAG_MSG"

# 推送标签
echo -e "${BLUE}🚀 推送标签到GitHub...${NC}"
git push origin "$TAG_NAME"

echo ""
echo -e "${GREEN}✅ 标签创建完成！${NC}"
echo ""
echo "📊 标签信息:"
echo "   名称: $TAG_NAME"
echo "   内容: $TAG_MSG"
echo "   时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "🔗 查看标签: git tag -l"
echo "🔗 查看详情: git show $TAG_NAME"
