// frontend/src/components/ForecastSlider.jsx
import React from 'react';
import styles from './ForecastSlider.module.css';

const ForecastSlider = ({ value, onChange, id }) => {
  return (
    <div className={styles.sliderContainer}>
      <input
        type="range"
        id={`${id}-range`}
        className={styles.rangeInput}
        min={10}
        max={200}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        aria-label="Выберите количество дней для прогноза"
        aria-valuemin={10}
        aria-valuemax={200}
        aria-valuenow={value}
      />
    </div>
  );
};

export default ForecastSlider;