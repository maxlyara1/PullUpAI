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
        max={365}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        aria-label="Выберите количество дней для прогноза"
        aria-valuemin={10}
        aria-valuemax={365}
        aria-valuenow={value}
      />
      <div className={styles.valueMarkers}>
        <span>10</span>
        <span>100</span>
        <span>200</span>
        <span>300</span>
        <span>365</span>
      </div>
    </div>
  );
};

export default ForecastSlider;