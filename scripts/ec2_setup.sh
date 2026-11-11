#!/bin/bash
# EC2(Ubuntu 24.04) 첫 부팅 시 자동 실행되는 설정 스크립트 (사용자 데이터)
# 1) 스왑 2GB 추가  2) Docker + Compose 설치  3) 코드 내려받기  4) 컨테이너 실행
# 진행 로그: /var/log/ec2_setup.log
set -euxo pipefail
exec > >(tee -a /var/log/ec2_setup.log) 2>&1

REPO_URL="https://github.com/ksumin020275/cloud-mlops-team00.git"
BRANCH="main"
APP_DIR="/home/ubuntu/app"
# repo 안에서 Dockerfile·docker-compose.yml이 있는 폴더 (repo 맨 위에 있으면 "." 그대로)
APP_SUBDIR="."

# 1) 스왑 2GB (t3.micro 메모리 1GB 보완, 이미지 빌드 중 메모리 부족 방지)
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# 2) Docker + Docker Compose 플러그인 설치
if ! command -v docker >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y ca-certificates curl git
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker
usermod -aG docker ubuntu        # ubuntu 사용자가 sudo 없이 docker 사용

# 3) 코드 내려받기 (이미 있으면 최신으로 갱신)
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
chown -R ubuntu:ubuntu "$APP_DIR"

# 4) 컨테이너 빌드·실행 (같은 프로젝트 이름이라 다시 실행해도 컨테이너가 중복 생성되지 않고 교체됨)
cd "$APP_DIR/$APP_SUBDIR"
docker compose up -d --build
docker compose ps
