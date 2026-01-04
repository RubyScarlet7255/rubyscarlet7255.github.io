#!/bin/bash

# RubyScarlet 的一键部署脚本
echo "🚀 开始构建..."
cd docs

echo "📦 安装依赖..."
yarn install

echo "🔨 构建站点..."
yarn build

echo "📁 创建必要文件..."
# 确保有 .nojekyll 文件
touch .vitepress/dist/.nojekyll
# 确保有 CNAME 文件
echo 'rubyscarlet7255.github.io' > .vitepress/dist/CNAME

echo "🚀 部署到 GitHub Pages..."
npx gh-pages -d .vitepress/dist --dotfiles

echo "✅ 部署完成！访问：https://rubyscarlet7255.github.io"
echo "⏰ 可能需要 1-2 分钟刷新"