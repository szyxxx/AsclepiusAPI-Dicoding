# Use the official Node.js image
FROM node:22

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of your application code
COPY . .

COPY tensorflow.dll node_modules/\@tensorflow/tfjs-node/lib/napi-v8/

COPY tensorflow.dll node_modules/\@tensorflow/tfjs-node/deps/lib/

# COPY tensorflow.dll node_modules/\@tensorflow/tfjs-node/lib/napi-v8/

# Expose the port your app runs on
EXPOSE 8080

# Command to run your application
CMD ["node", "index.js"]
