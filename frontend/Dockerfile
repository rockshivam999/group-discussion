FROM node:18-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm install

ARG VITE_BACKEND_HOST=localhost
ENV VITE_BACKEND_HOST=${VITE_BACKEND_HOST}

COPY vite.config.js index.html ./
COPY src ./src

RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
