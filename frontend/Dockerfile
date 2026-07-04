# Step 1: Build Vue app
FROM node:22 AS build-stage

ARG VITE_PORTFOLIO_PASSWORD
ENV VITE_PORTFOLIO_PASSWORD=${VITE_PORTFOLIO_PASSWORD}

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN echo "VITE_PORTFOLIO_PASSWORD=${VITE_PORTFOLIO_PASSWORD}" > .env && npm run build

# Step 2: Serve with Nginx
FROM nginx:stable-alpine AS production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
