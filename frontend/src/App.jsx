import React, { useState, useEffect, useCallback } from 'react';
import { Routes, Route } from 'react-router-dom'; // Import routing components
import AddDataForm from './components/AddDataForm';
import DataTable from './components/DataTable';
import Chart from './components/Chart';
import WeightCategorySelector from './components/WeightCategorySelector';
import ForecastSlider from './components/ForecastSlider';
import * as api from './services/api';
import styles from './styles/App.module.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChartLine, faDownload, faUndo, faMoon, faSun } from '@fortawesome/free-solid-svg-icons';
import { format } from 'date-fns';
import ru from 'date-fns/locale/ru';

// Error Boundary Component
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        console.error("ErrorBoundary caught an error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return <h1>Что-то пошло не так.</h1>;
        }
        return this.props.children;
    }
}

function App() {
    // Загрузка начальных значений из localStorage или использование значений по умолчанию
    const getSavedValue = (key, defaultValue) => {
        const saved = localStorage.getItem(key);
        return saved !== null ? JSON.parse(saved) : defaultValue;
    };

    const [data, setData] = useState([]);
    const [chart2Data, setChart2Data] = useState(null);
    const [weightCategory, setWeightCategory] = useState(() => getSavedValue('weightCategory', 'до 75'));
    const [forecastDays, setForecastDays] = useState(() => getSavedValue('forecastDays', 90));
    const [availableWeightCategories, setAvailableWeightCategories] = useState([]);
    const [achievementDates, setAchievementDates] = useState({});
    const [pullupStandards, setPullupStandards] = useState({});
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);
    const [isInitialLoad, setIsInitialLoad] = useState(true);
    const [isDarkMode, setIsDarkMode] = useState(() => getSavedValue('isDarkMode', true));

    // Сохранение значений в localStorage при их изменении
    const handleSetWeightCategory = (value) => {
        setWeightCategory(value);
        localStorage.setItem('weightCategory', JSON.stringify(value));
        // Вызываем обновление нормативов сразу после изменения весовой категории
        updateData();
    };

    const handleSetForecastDays = (value) => {
        setForecastDays(value);
        localStorage.setItem('forecastDays', JSON.stringify(value));
    };

    const toggleDarkMode = () => {
        const newValue = !isDarkMode;
        setIsDarkMode(newValue);
        localStorage.setItem('isDarkMode', JSON.stringify(newValue));
    };

    useEffect(() => {
        const link1 = document.createElement('link');
        link1.rel = 'preconnect';
        link1.href = 'https://fonts.googleapis.com';
        document.head.appendChild(link1);

        const link2 = document.createElement('link');
        link2.rel = 'preconnect';
        link2.href = 'https://fonts.gstatic.com';
        link2.crossOrigin = 'true';
        document.head.appendChild(link2);

        const link3 = document.createElement('link');
        link3.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap';
        link3.rel = 'stylesheet';
        document.head.appendChild(link3);

        return () => {
            document.head.removeChild(link1);
            document.head.removeChild(link2);
            document.head.removeChild(link3);
        };
    }, []);

    const updateData = useCallback(async () => {
        setLoading(true);
        try {
            console.log(`Fetching data for: weight=${weightCategory}, days=${forecastDays}`);
            const result = await api.getPredictionData(weightCategory, forecastDays);
            setData(result.data_2025);
            // Парсим JSON-строки из бэкенда
            setChart2Data(JSON.parse(result.chart2));
            setAchievementDates(result.achievement_dates);
            setPullupStandards(result.pullup_standards);
            
            console.log('Updated standards for weight category:', weightCategory);
            console.log('Achievement dates:', result.achievement_dates);
            console.log('Pullup standards:', result.pullup_standards);

            setError('');
        } catch (error) {
            console.error("Error fetching data:", error);
            setError(error.message || 'Произошла ошибка при загрузке данных.');
        } finally {
            setLoading(false);
            if (isInitialLoad) setIsInitialLoad(false);
        }
    }, [weightCategory, forecastDays, isInitialLoad]);

    useEffect(() => {
        updateData();

        const fetchCategories = async () => {
            try {
                const standards = await api.getOriginalStandards();
                setAvailableWeightCategories([...new Set(standards.map(item => item.weight_category))]);
            } catch (error) {
                console.error("Error fetching categories:", error);
                setError(error.message || 'Произошла ошибка при загрузке весовых категорий.');
            }
        };
        fetchCategories();
    }, [updateData]);

    useEffect(() => {
        document.body.classList.toggle(styles.darkMode, isDarkMode);
    }, [isDarkMode]);

    const handleAddData = async (newData) => {
        try {
            await api.addData(newData);
            await updateData();
        } catch (error) {
            console.error("Error adding data:", error);
            setError(error.message || 'Произошла ошибка при добавлении данных.');
        }
    };

    const handleDeleteData = async (index) => {
        try {
            await api.deleteData(index);
            await updateData();
        } catch (error) {
            console.error("Error deleting data:", error);
            setError(error.message || 'Произошла ошибка при удалении данных.');
        }
    };

    const handleResetData = async () => {
        // Добавляем диалог подтверждения
        const confirmReset = window.confirm('Вы действительно хотите сбросить все данные? Это действие нельзя отменить.');
        if (!confirmReset) return;
        
        try {
            await api.resetData();
            await updateData();
        } catch (error) {
            console.error("Error resetting data:", error);
            setError(error.message || 'Произошла ошибка при сбросе данных.');
        }
    };

    return (
        <div className={`${styles.container} ${isDarkMode ? styles.darkMode : ""}`}>
            <header className={styles.header}>
                <div className={styles.headerLeft}>
                    <h1 className={styles.title}>
                        <FontAwesomeIcon icon={faChartLine} className={styles.icon} />
                        PullUpAI
                    </h1>
                    <div className={styles.headerControls}>
                        <div className={styles.controlGroup}>
                            <label className={styles.controlLabel}>Вес:</label>
                            <WeightCategorySelector
                                categories={availableWeightCategories}
                                selectedCategory={weightCategory}
                                onCategoryChange={handleSetWeightCategory}
                                className={styles.controlSelect}
                            />
                        </div>
                        <div className={styles.controlGroup}>
                            <label className={styles.controlLabel}>Прогноз:</label>
                            <ForecastSlider
                                value={forecastDays}
                                onChange={handleSetForecastDays}
                                id="forecast-days"
                            />
                            <span className={styles.controlValue}>{forecastDays}</span>
                        </div>
                    </div>
                </div>
                <div className={styles.headerRight}>
                    <button onClick={toggleDarkMode} className={styles.themeToggle} aria-label="Переключить тему">
                        <FontAwesomeIcon icon={isDarkMode ? faSun : faMoon} />
                    </button>
                </div>
            </header>

            {error && <p className={styles.errorMessage}>{error}</p>}
            {loading && isInitialLoad && <p className={styles.loadingMessage}>Загрузка...</p>}

            <Routes>
                <Route path="/" element={
                    <div className={styles.gridContainer}>
                        {/* Левая колонка - таблицы и данные */}
                        <div className={styles.mainContent}>
                            <div className={`${styles.tableWrapper} ${styles.dataInputWrapper}`}>
                                <h2 className={styles.subtitle}>Добавление тренировки</h2>
                                <AddDataForm onAddData={handleAddData} />
                            </div>
                            <div className={`${styles.tableWrapper} ${styles.narrowTable}`}>
                                <h2 className={styles.subtitle}>Журнал тренировок</h2>
                                <DataTable data={data} onDeleteData={handleDeleteData} updateData={updateData} />
                                <div className={styles.tableFooter}>
                                    <button onClick={handleResetData} className={styles.tableButton} aria-label="Сбросить данные">
                                        <FontAwesomeIcon icon={faUndo} className={styles.buttonIcon} />
                                        Сбросить
                                    </button>
                                    <button href="/api/download" download="data_2025.csv" className={styles.tableButton} aria-label="Скачать CSV">
                                        <FontAwesomeIcon icon={faDownload} className={styles.buttonIcon} />
                                        Скачать CSV
                                    </button>
                                </div>
                            </div>
                            <div className={styles.tableWrapper}>
                                <h2 className={styles.subtitle}>
                                    Прогноз Достижения Разрядов
                                </h2>
                                <table className={styles.achievementsTable}>
                                    <thead>
                                        <tr>
                                            <th>Разряд</th>
                                            <th>Подтягивания<br/>с весом</th>
                                            <th>Прогноз<br/>даты</th>
                                            <th>Статус</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(achievementDates).map(([rank, days]) => {
                                            let predictedDate = "Не достигнуто в прогнозе";
                                            let achievementEmoji = "🎯";
                                            
                                            if (typeof days === "number") {
                                                const startDate = data && data.length > 0 
                                                    ? new Date(data[0].date) 
                                                    : new Date(new Date().getFullYear() + 1, 0, 14);
                                                let date = new Date(startDate);
                                                date.setDate(startDate.getDate() + days);
                                                predictedDate = format(date, "dd.MM.yyyy", { locale: ru });
                                                
                                                // Проверяем, достигнут ли разряд
                                                const today = new Date();
                                                if (date <= today) {
                                                    achievementEmoji = "🏆";
                                                }
                                            }

                                            // Выбираем эмодзи для разряда
                                            let rankEmoji = "";
                                            switch(rank) {
                                                case "МСМК":
                                                    rankEmoji = "🥇";
                                                    break;
                                                case "МС":
                                                    rankEmoji = "🥈";
                                                    break;
                                                case "КМС":
                                                    rankEmoji = "🥉";
                                                    break;
                                                case "I":
                                                    rankEmoji = "⭐";
                                                    break;
                                                case "II":
                                                    rankEmoji = "⭐";
                                                    break;
                                                case "III":
                                                    rankEmoji = "⭐";
                                                    break;
                                                default:
                                                    rankEmoji = "🎯";
                                            }
                                            
                                            return (
                                                <tr key={rank}>
                                                    <td>{rankEmoji} <span>{rank}</span></td>
                                                    <td>{pullupStandards[rank]}</td>
                                                    <td>{predictedDate}</td>
                                                    <td>{achievementEmoji}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Правая колонка - график и пояснения */}
                        <div className={styles.chartsColumn}>
                            {chart2Data && (
                                <>
                                    <div className={styles.chartWrapper}>
                                        <ErrorBoundary>
                                            <Chart
                                                data={chart2Data.data}
                                                standards={chart2Data.standards}
                                                title={chart2Data.title}
                                                xAxisLabel={chart2Data.xAxisLabel}
                                                yAxisLabel={chart2Data.yAxisLabel}
                                                darkMode={isDarkMode}
                                            />
                                        </ErrorBoundary>
                                    </div>
                                    <div className={styles.explainerBox}>
                                        <h3>🚀 Интеллектуальный прогноз прогресса</h3>
                                        
                                        <div className={`${styles.explainerSection} ${styles.twoColumns}`}>
                                            <div className={styles.column}>
                                                <h4>📊 Линии на графике</h4>
                                                <p className={styles.colorLine}>
                                                    <span className={`${styles.colorDot} ${styles.blueDot}`}></span>
                                                    <strong>Синяя</strong> — средние подтягивания за тренировку
                                                </p>
                                                <p className={styles.colorLine}>
                                                    <span className={`${styles.colorDot} ${styles.greenDot}`}></span>
                                                    <strong>Зелёная</strong> — максимум без дополнительного веса
                                                </p>
                                                <p className={styles.colorLine}>
                                                    <span className={`${styles.colorDot} ${styles.purpleDot}`}></span>
                                                    <strong>Фиолетовая</strong> — максимум с весом 24 кг
                                                </p>
                                                <p className={styles.colorLine}>
                                                    <span className={`${styles.colorDot} ${styles.redDot}`}></span>
                                                    <strong>Красные точки</strong> — фактические результаты
                                                </p>
                                            </div>
                                            <div className={styles.column}>
                                                <h4>⚙️ Как работает прогноз</h4>
                                                <p>Алгоритм анализирует динамику ваших результатов и строит регрессионную модель, учитывая:</p>
                                                <ul className={styles.factorsList}>
                                                    <li>Частоту и регулярность тренировок</li>
                                                    <li>Динамику прироста результатов</li>
                                                    <li>Скорость прогресса в похожих весовых категориях</li>
                                                </ul>
                                            </div>
                                        </div>
                                        
                                        <div className={styles.explainerSection}>
                                            <div className={styles.infoGrid}>
                                            <div>
                                                    <strong>🎯</strong> — в процессе достижения
                                                </div>
                                                
                                                <div>
                                                    <span className={styles.lineIcon}>|</span>
                                                    <strong>Вертикальные линии</strong> — прогноз достижения разряда
                                                </div>
                                                
                                                <div>
                                                    <strong>🏆</strong> — норматив выполнен
                                                </div>
                                                <div>
                                                    <span className={styles.lineIcon}>—</span>
                                                    <strong>Горизонтальные линии</strong> — нормативы разрядов
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                } />
            </Routes>
        </div>
    );
}

export default App;