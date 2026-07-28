FROM oven/bun:1 AS base
WORKDIR /usr/src/app

FROM base AS install
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

FROM base AS frontend-build
COPY --from=install /usr/src/app/node_modules node_modules
COPY . .
ENV NODE_ENV=production
RUN bun run build