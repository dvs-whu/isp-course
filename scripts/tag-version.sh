#!/bin/bash
# 智能信号处理课程版本标签脚本
# 用法: ./scripts/tag-version.sh [version] [message]

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 切换到vault目录
VAULT_DIR="/Users/leiyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/智能信号处理"
cd "$VAULT_DIR"

echo -e "${BLUE}🏷️  智能信号处理课程版本标签工具${NC}"
echo "======================================"
echo ""

# 显示现有版本标签
echo "📋 现有版本标签:"
git tag -l "v*" --sort=-v:refname | head -10
echo ""

# 获取版本号
if [[ -n "$1" ]]; then
    VERSION="$1"
else
    read -p "📌 请输入版本号 (如 v1.1.0): " VERSION
fi

# 验证版本号格式
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ 无效的版本号格式，请使用 vX.Y.Z 格式"
    exit 1
fi

# 获取版本说明
if [[ -n "$2" ]]; then
    VERSION_MSG="$2"
else
    echo ""
    echo "📝 请输入版本说明:"
    read VERSION_MSG
fi

if [[ -z "$VERSION_MSG" ]]; then
    echo "❌ 版本说明不能为空"
    exit 1
fi

# 检查标签是否已存在
if git tag -l "$VERSION" | grep -q "$VERSION"; then
    echo -e "${YELLOW}⚠️  标签 $VERSION 已存在${NC}"
    read -p "是否覆盖？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 操作取消"
        exit 1
    fi
    git tag -d "$VERSION"
fi

# 创建标签
echo ""
echo -e "${BLUE}🏷️  创建版本标签...${NC}"
git tag -a "$VERSION" -m "$VERSION_MSG"

# 推送标签
echo -e "${BLUE}🚀 推送标签到GitHub...${NC}"
git push origin "$VERSION"

echo ""
echo -e "${GREEN}✅ 版本标签创建完成！${NC}"
echo ""
echo "📊 版本信息:"
echo "   版本: $VERSION"
echo "   说明: $VERSION_MSG"
echo "   时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "🔗 查看版本: git tag -l 'v*'"
echo "🔗 查看详情: git show $VERSION"
echo "🔗 GitHub发布页: https://github.com/YOUR_USERNAME/isp-course/releases"
