// frontend/src/services/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

class ApiError extends Error {
    constructor(status, message) {
        super(message);
        this.status = status;
        this.name = 'ApiError';
    }
}

//  Функция для получения данных для прогноза
export const getPredictionData = async (weightCategory, forecastDays) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/prediction`, {
            params: { weight_category: weightCategory, forecast_days: forecastDays },
        });
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new ApiError(
                error.response.status,
                error.response.data.message || 'Произошла ошибка при получении данных для прогноза.'
            );
        } else if (error.request) {
            throw new ApiError(
                0,
                'Не удалось соединиться с сервером. Проверьте подключение к интернету.'
            );
        } else {
            throw new ApiError(
                -1,
                'Произошла неизвестная ошибка при получении данных для прогноза.'
            );
        }
    }
};

//  Функция для добавления данных
export const addData = async (data) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/data`, data);
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new ApiError(
                error.response.status,
                error.response.data.message || 'Произошла ошибка при добавлении данных.'
            );
        } else if (error.request) {
            throw new ApiError(
                0,
                'Не удалось соединиться с сервером. Проверьте подключение к интернету.'
            );
        } else {
            throw new ApiError(
                -1,
                'Произошла неизвестная ошибка при добавлении данных.'
            );
        }
    }
};

//  Функция для обновления данных
export const updateData = async (index, updatedData) => {
    try {
        const response = await axios.put(`${API_BASE_URL}/data/${index}`, updatedData);
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new ApiError(
                error.response.status,
                error.response.data.message || 'Произошла ошибка при обновлении данных.'
            );
        } else if (error.request) {
            throw new ApiError(
                0,
                'Не удалось соединиться с сервером. Проверьте подключение к интернету.'
            );
        } else {
            throw new ApiError(
                -1,
                'Произошла неизвестная ошибка при обновлении данных.'
            );
        }
    }
};

//  Функция для удаления данных
export const deleteData = async (index) => {
    try {
        const response = await axios.delete(`${API_BASE_URL}/data/${index}`);
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new ApiError(
                error.response.status,
                error.response.data.message || 'Произошла ошибка при удалении данных.'
            );
        } else if (error.request) {
            throw new ApiError(
                0,
                'Не удалось соединиться с сервером. Проверьте подключение к интернету.'
            );
        } else {
            throw new ApiError(
                -1,
                'Произошла неизвестная ошибка при удалении данных.'
            );
        }
    }
};

//  Функция для сброса данных
export const resetData = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/reset`);
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new ApiError(
                error.response.status,
                error.response.data.message || 'Произошла ошибка при сбросе данных.'
            );
        } else if (error.request) {
            throw new ApiError(
                0,
                'Не удалось соединиться с сервером. Проверьте подключение к интернету.'
            );
        } else {
            throw new ApiError(
                -1,
                'Произошла неизвестная ошибка при сбросе данных.'
            );
        }
    }
};

//  Функция для получения оригинальных нормативов
export const getOriginalStandards = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/original-standards`);
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new ApiError(
                error.response.status,
                error.response.data.message || 'Произошла ошибка при получении нормативов.'
            );
        } else if (error.request) {
            throw new ApiError(
                0,
                'Не удалось соединиться с сервером. Проверьте подключение к интернету.'
            );
        } else {
            throw new ApiError(
                -1,
                'Произошла неизвестная ошибка при получении нормативов.'
            );
        }
    }
};

// Функция для получения всей истории
export const getAllData = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/history`);
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new ApiError(
                error.response.status,
                error.response.data.message || 'Произошла ошибка.'
            );
        } else if (error.request) {
            throw new ApiError(
                0,
                'Не удалось соединиться с сервером. Проверьте подключение к интернету.'
            );
        } else {
            throw new ApiError(
                -1,
                'Произошла неизвестная ошибка.'
            );
        }
    }
};