// frontend/src/components/AddDataForm.jsx
import React, { useState, useRef, useEffect } from 'react';
import styles from './AddDataForm.module.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlus } from '@fortawesome/free-solid-svg-icons';

const AddDataForm = ({ onAddData }) => {
    const [date, setDate] = useState(() => {
        const today = new Date();
        return today.toISOString().substring(0, 10);
    });
    const [avg, setAvg] = useState('');
    const [sum, setSum] = useState('');
    const avgInputRef = useRef(null);

    useEffect(() => {
        // Устанавливаем фокус при первой загрузке
        if (avgInputRef.current) {
            avgInputRef.current.focus();
        }
    }, []);

    // Обработка изменения суммы
    const handleSumChange = (e) => {
        const value = e.target.value;
        setSum(value);
        if (value && !isNaN(Number(value))) {
            // Рассчитываем среднее из суммы (делим на 5 подходов)
            setAvg((Number(value) / 5).toFixed(1));
        } else {
            setAvg('');
        }
    };

    // Обработка изменения среднего
    const handleAvgChange = (e) => {
        const value = e.target.value;
        setAvg(value);
        if (value && !isNaN(Number(value))) {
            // Рассчитываем сумму из среднего (умножаем на 5 подходов)
            setSum(Math.round(Number(value) * 5).toString());
        } else {
            setSum('');
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if ((avg.trim() === '' && sum.trim() === '') || 
            (avg.trim() !== '' && isNaN(Number(avg))) ||
            (sum.trim() !== '' && isNaN(Number(sum)))) {
            alert('Пожалуйста, введите корректное среднее значение или сумму подтягиваний.');
            return;
        }

        const newData = {
            date,
            avg_pullups: avg.trim() !== '' ? Number(avg) : null,
            total_pullups: sum.trim() !== '' ? Number(sum) : null,
        };

        onAddData(newData);
        // Сбрасываем только значения, но сохраняем текущую дату
        setAvg('');
        setSum('');
        
        // Фокус на поле ввода после добавления
        if (avgInputRef.current) {
            avgInputRef.current.focus();
        }
    };

    const handleKeyDown = (e) => {
        // Добавляем возможность отправить форму по Enter
        if (e.key === 'Enter' && 
            ((avg.trim() !== '' && !isNaN(Number(avg))) || 
             (sum.trim() !== '' && !isNaN(Number(sum))))) {
            handleSubmit(e);
        }
    };

    return (
        <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.formGrid}>
                <div className={styles.formGroup}>
                    <label htmlFor="date" className={styles.label}>Дата:</label>
                    <input
                        type="date"
                        id="date"
                        value={date}
                        onChange={(e) => setDate(e.target.value)}
                        className={styles.input}
                        required
                    />
                </div>
                <div className={styles.formGroup}>
                    <label htmlFor="avg" className={styles.label}>Среднее (5 подходов):</label>
                    <input
                        ref={avgInputRef}
                        type="number"
                        id="avg"
                        value={avg}
                        onChange={handleAvgChange}
                        onKeyDown={handleKeyDown}
                        className={styles.input}
                        placeholder="0.0"
                        step="0.1"
                        min="0"
                        required
                    />
                </div>
                <div className={styles.formGroup}>
                    <label htmlFor="sum" className={styles.label}>Сумма (5 подходов):</label>
                    <input
                        type="number"
                        id="sum"
                        value={sum}
                        onChange={handleSumChange}
                        onKeyDown={handleKeyDown}
                        className={styles.input}
                        placeholder="0"
                        step="1"
                        min="0"
                    />
                </div>
                <button type="submit" className={styles.button}>
                    <FontAwesomeIcon icon={faPlus} className={styles.buttonIcon} />
                    Добавить
                </button>
            </div>
        </form>
    );
};

export default AddDataForm;