#!/bin/bash
# EC2 서버에서 실행하는 배포 스크립트
# GitHub Actions가 SSH로 실행하고, 서버에 접속해 직접 실행해도 된다: bash ~/app/scripts/deploy.sh
#   1) main 최신 코드로 갱신  2) 이미지 다시 빌드 후 컨테이너 교체  3) healthy 상태까지 확인
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/app}"
BRANCH="${BRANCH:-main}"
CONTAINER="bike-rental-app"

cd "$APP_DIR"
echo "== 배포 전 버전: $(git rev-parse --short HEAD)"

# 1) 최신 코드 받기 (서버에서 직접 고친 내용은 버리고 GitHub와 똑같이 맞춤)
git fetch origin "$BRANCH"
git reset --hard "origin/$BRANCH"
echo "== 배포할 버전: $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"

# 2) 이미지 다시 빌드 후 컨테이너 교체
#    같은 프로젝트 이름(bike-rental)·컨테이너 이름이라 새 컨테이너가 기존 것을 대체한다 (중복 실행 없음)
docker compose up -d --build --remove-orphans
docker image prune -f            # 이전 빌드에서 남은 이름 없는 이미지 정리 (디스크 확보)

# 3) 헬스체크 통과까지 최대 3분 대기
for i in $(seq 1 36); do
  status=$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "none")
  echo "  [$i] health: $status"
  if [ "$status" = "healthy" ]; then
    docker compose ps
    echo "== 배포 성공"
    exit 0
  fi
  sleep 5
done

echo "== 배포 실패: 컨테이너가 healthy 상태가 되지 않았습니다"
docker compose ps
docker compose logs --tail 50
exit 1
