// frontend/src/components/DataTable.jsx
import React, { useState } from 'react';
import DatePicker from "react-datepicker";
import { format } from 'date-fns';
import ru from 'date-fns/locale/ru';
import styles from './DataTable.module.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEdit, faTrash, faSave, faTimes, faChevronLeft, faChevronRight } from '@fortawesome/free-solid-svg-icons';
import * as api from '../services/api';

function DataTable({ data, onDeleteData, updateData }) {
    const [editIndex, setEditIndex] = useState(null);
    const [editedDate, setEditedDate] = useState(null);
    const [editedAvgPullups, setEditedAvgPullups] = useState('');
    const [error, setError] = useState(''); // Добавили состояние для ошибки
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10; // Количество записей на странице

    const handleEdit = (index, date, avg_pullups) => {
        setEditIndex(index);
        setEditedDate(new Date(date));
        setEditedAvgPullups(avg_pullups);
        setError(''); // Сбрасываем ошибку при начале редактирования
    };

    const handleSave = async (index) => {
        try {
            // Валидация перед сохранением
            if (!editedDate) {
                setError('Выберите дату.');
                return;
            }
            if (isNaN(parseFloat(editedAvgPullups)) || parseFloat(editedAvgPullups) < 0) {
                setError('Введите корректное среднее количество подтягиваний.');
                return;
            }
             if (parseFloat(editedAvgPullups) > 100) {
              setError('Значение не может быть больше 100');
              return
            }

            const updatedData = {
                date: format(editedDate, 'yyyy-MM-dd'),
                avg_pullups: parseFloat(editedAvgPullups),
                total_pullups: null, //  Предполагаем, что total_pullups не редактируется
            };
            await api.updateData(index, updatedData);
            setEditIndex(null);
            setEditedDate(null);
            setEditedAvgPullups('');
            setError(''); // Сбрасываем ошибку
            await updateData();
        } catch (error) {
            console.error("Ошибка при обновлении данных:", error);
            setError(error.message || "Произошла ошибка при сохранении."); // Показываем сообщение об ошибке
        }
    };

    const handleCancel = () => {
        setEditIndex(null);
        setEditedDate(null);
        setEditedAvgPullups('');
        setError(''); // Сбрасываем ошибку
    };

    // Пагинация
    const totalPages = Math.ceil(data.length / itemsPerPage);
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    const currentItems = data.slice(indexOfFirstItem, indexOfLastItem);

    const paginate = (pageNumber) => setCurrentPage(pageNumber);
    const nextPage = () => setCurrentPage(prev => prev < totalPages ? prev + 1 : prev);
    const prevPage = () => setCurrentPage(prev => prev > 1 ? prev - 1 : prev);

    return (
        <div>
            <div className={styles.tableWrapper}>
                <table className={styles.table}>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Дата</th>
                            <th>Среднее число подтягиваний</th>
                            <th>Действие</th>
                        </tr>
                    </thead>
                    <tbody>
                        {currentItems.map((item, localIndex) => {
                            const index = indexOfFirstItem + localIndex;
                            return (
                                <tr key={index}>
                                    <td>{index + 1}</td>
                                    <td>
                                        {editIndex === index ? (
                                            <DatePicker
                                                selected={editedDate}
                                                onChange={(newDate) => setEditedDate(newDate)}
                                                dateFormat="dd.MM.yyyy"
                                                locale={ru}
                                                placeholderText="дд.мм.гггг"
                                                className={styles.editInput}
                                            />
                                        ) : (
                                            format(new Date(item.date), 'dd.MM.yyyy')
                                        )}
                                    </td>
                                    <td className={styles.numberCell}>
                                        {editIndex === index ? (
                                            <input
                                                type="number"
                                                value={editedAvgPullups}
                                                onChange={(e) => setEditedAvgPullups(e.target.value)}
                                                className={styles.editInput}
                                            />
                                        ) : (
                                            item.avg_pullups?.toFixed(1)
                                        )}
                                    </td>
                                    <td>
                                        {editIndex === index ? (
                                            <>
                                                <button onClick={() => handleSave(index)} className={styles.button} aria-label="Сохранить">
                                                    <FontAwesomeIcon icon={faSave} className={styles.buttonIcon} />
                                                </button>
                                                <button onClick={handleCancel} className={styles.button} aria-label="Отменить">
                                                    <FontAwesomeIcon icon={faTimes} className={styles.buttonIcon} />
                                                </button>
                                            </>
                                        ) : (
                                            <>
                                                <button onClick={() => handleEdit(index, item.date, item.avg_pullups)} className={styles.button} aria-label="Редактировать">
                                                    <FontAwesomeIcon icon={faEdit} className={styles.buttonIcon} />
                                                </button>
                                                <button onClick={() => onDeleteData(index)} className={styles.button} aria-label="Удалить">
                                                    <FontAwesomeIcon icon={faTrash} className={styles.buttonIcon} />
                                                </button>
                                            </>
                                        )}
                                    </td>
                                </tr>
                            );
                        })}
                        {currentItems.length === 0 && (
                            <tr>
                                <td colSpan="4" className={styles.noData}>
                                    Нет данных для отображения
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
            
            {data.length > itemsPerPage && (
                <div className={styles.pagination}>
                    <button 
                        className={styles.paginationButton} 
                        onClick={prevPage}
                        disabled={currentPage === 1}
                    >
                        <FontAwesomeIcon icon={faChevronLeft} />
                    </button>
                    
                    {[...Array(totalPages).keys()].map(number => (
                        <button
                            key={number + 1}
                            onClick={() => paginate(number + 1)}
                            className={`${styles.paginationButton} ${currentPage === number + 1 ? styles.currentPage : ''}`}
                        >
                            {number + 1}
                        </button>
                    ))}
                    
                    <button 
                        className={styles.paginationButton} 
                        onClick={nextPage}
                        disabled={currentPage === totalPages}
                    >
                        <FontAwesomeIcon icon={faChevronRight} />
                    </button>
                </div>
            )}
            
            {error && <p className={styles.error}>{error}</p>} {/* Отображаем сообщение об ошибке */}
        </div>
    );
}

export default DataTable;