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
                    Хотите узнать, как быстро вы сможете увеличить количество подтягиваний? Этот инструмент поможет вам
                    спрогнозировать ваш прогресс в тренировках.  Он анализирует ваши прошлые результаты, чтобы показать,
                    чего можно ожидать в будущем.
                </p>
            </section>

            <section className={styles.section}>
                <h2 className={styles.subtitle}>Как строится прогноз?</h2>
                <p>
                    Прогноз использует ваши данные о тренировках, чтобы понять, как вы прогрессируете.  Вот что учитывается:
                </p>
                <ul>
                    <li><strong>Дата тренировки:</strong> Когда проходила ваша тренировка.</li>
                    <li><strong>Среднее количество подтягиваний:</strong> Сколько подтягиваний в среднем за подход вы делаете.</li>
                    <li><strong>Ваша весовая категория:</strong>  Это помогает точнее настроить прогноз.</li>
                    <li><strong>На сколько дней нужен прогноз:</strong> Вы сами выбираете, на какой срок заглянуть в будущее.</li>
                    <li><strong>Метод прогноза:</strong> Можно выбрать простой или более сложный способ расчета, чтобы увидеть разные варианты развития.</li>
                </ul>
                <p>
                    Инструмент анализирует эти данные и рисует линию вашего прогресса. Затем он просто продолжает эту линию в будущее,
                    чтобы показать, какими могут быть ваши результаты, если вы продолжите тренироваться в том же темпе.
                </p>
            </section>

            <section className={styles.section}>
                <h2 className={styles.subtitle}>Важные детали прогноза</h2>
                <p>
                  В расчетах используются специальные коэффициенты, чтобы сделать прогноз точнее:
                </p>
                <ul>
                  <li>
                    <strong>k1: Отношение среднего к максимуму.</strong>  Этот коэффициент помогает понять, сколько раз вы максимально можете подтянуться,
                    исходя из вашего среднего результата за тренировку.  Для расчета k1 используется специальная модель, но если она недоступна,
                    используется стандартное значение.
                  </li>
                  <li>
                    <strong>k2: Учет веса.</strong>  Этот коэффициент важен, если вы планируете подтягиваться с дополнительным весом. Он учитывает ваш собственный вес
                    и вес отягощения, чтобы прогноз был более реалистичным для тренировок с весом.
                  </li>
                </ul>
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