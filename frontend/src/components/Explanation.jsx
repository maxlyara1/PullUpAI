// frontend/src/components/Explanation.jsx
import React from 'react';
import styles from '../styles/Explanation.module.css'; // Import the CSS module

function Explanation() {
    return (
        <div className={styles.container}>
            <h1 className={styles.title}>Как работает прогноз подтягиваний</h1>

            <section className={styles.section}>
                <h2 className={styles.subtitle}>Зачем нужен прогноз?</h2>
                <p>
                    Хотите узнать, как быстро можно увеличить количество подтягиваний? Этот инструмент анализирует данные ваших тренировок и показывает, чего можно ожидать в будущем.
                </p>
            </section>

            <section className={styles.section}>
                <h2 className={styles.subtitle}>Как строится прогноз?</h2>
                <p>
                    Прогноз использует следующие входные данные:
                </p>
                <ul>
                    <li><strong>Дата тренировки:</strong> когда проходила ваша тренировка.</li>
                    <li><strong>Среднее количество подтягиваний:</strong> результат за 5 подходов.</li>
                    <li><strong>Результаты предыдущих тренировок:</strong> данные за последние 3 тренировки.</li>
                </ul>
                <p>
                    Коэффициенты модели не интерпретируемы, поскольку применяется стандартизация (StandardScaler). Поэтому в пояснениях указываются только используемые фичи.
                </p>
            </section>

            <section className={styles.section}>
                <h2 className={styles.subtitle}>Спортивные разряды</h2>
                <p>
                    Приложение не только прогнозирует ваш прогресс, но и показывает, когда вы сможете достичь новых спортивных разрядов по подтягиваниям!
                    Оно использует официальные нормативы для разных весовых категорий, как для подтягиваний без веса, так и с отягощением,
                    чтобы вы могли ставить перед собой конкретные цели и видеть, как прогноз помогает вам их достигать.
                </p>
            </section>
        </div>
    );
}

export default Explanation;