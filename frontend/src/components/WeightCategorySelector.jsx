// frontend/src/components/WeightCategorySelector.jsx

import React from 'react';
import styles from './WeightCategorySelector.module.css'

function WeightCategorySelector({ categories, selectedCategory, onCategoryChange, id }) {
  const handleChange = (e) => {
    const newValue = e.target.value;
    // Вызываем обработчик изменений сразу с новым значением
    onCategoryChange(newValue);
  };
  
  return (
    <select
      value={selectedCategory}
      onChange={handleChange}
      className={styles.select}
      id={id}
    >
      {categories.map((category) => (
        <option key={category} value={category}>
          {category}
        </option>
      ))}
    </select>
  );
}

export default WeightCategorySelector;