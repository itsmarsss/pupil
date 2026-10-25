FROM apify/actor-node:22

COPY package*.json ./
RUN npm --quiet set progress=false && npm install --omit=dev

COPY . ./
RUN npm install --include=dev && npm run build && npm prune --omit=dev

CMD npm start
