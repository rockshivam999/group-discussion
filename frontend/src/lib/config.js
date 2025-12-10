const envHost = import.meta.env.VITE_BACKEND_HOST;
// Prefer window host in browser; fall back to env when provided and not "backend"
const backendHost = envHost && envHost !== "backend" ? envHost : window.location.hostname;
const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
const httpProtocol = window.location.protocol === "https:" ? "https" : "http";

export const wsBase = `${wsProtocol}://${backendHost}:8000`;
export const apiBase = `${httpProtocol}://${backendHost}:8000`;
