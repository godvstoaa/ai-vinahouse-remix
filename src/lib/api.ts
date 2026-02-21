import axios from 'axios';

const API_BASE = 'http://localhost:8001';

export const api = axios.create({
    baseURL: API_BASE,
    timeout: 300000,
    headers: {
        'Content-Type': 'application/json',
    },
});

export default api;