import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer
} from 'recharts';
import styles from './Chart.module.css';

const CustomTooltip = ({ active, payload, label, theme }) => {
  if (!active || !payload || !payload.length) return null;

  const getColor = (dataKey) => {
    switch (dataKey) {
      case 'average': return '#3699ff';
      case 'maximum': return '#0bb783';
      case 'withWeight': return '#8950fc';
      case 'actual': return '#f64e60';
      default: return theme.textColor;
    }
  };

  const getLabel = (dataKey) => {
    switch (dataKey) {
      case 'average': return 'Среднее';
      case 'maximum': return 'Максимум';
      case 'withWeight': return 'С весом';
      case 'actual': return 'Факт';
      default: return dataKey;
    }
  };

  return (
    <div className={styles.customTooltip} style={{
      backgroundColor: theme.tooltipBackground,
      borderColor: theme.tooltipBorder
    }}>
      {payload.map((entry, index) => {
        if (entry.value !== undefined && entry.value !== null) {
          return (
            <p key={index} style={{ color: getColor(entry.dataKey) }}>
              {`${getLabel(entry.dataKey)}: ${Number(entry.value).toFixed(1)}`}
            </p>
          );
        }
        return null;
      })}
      <p className={styles.tooltipDate}>{label.split('-').reverse().join('.')}</p>
    </div>
  );
};

const ChartComponent = ({ data, standards, title, xAxisLabel, yAxisLabel, darkMode = true, noUserData = false, message = '' }) => {
  const theme = {
    backgroundColor: darkMode ? '#1e293b' : '#ffffff',
    textColor: darkMode ? '#f8fafc' : '#1e293b',
    gridColor: darkMode ? '#334155' : '#e2e8f0',
    tooltipBackground: darkMode ? '#1e293b' : '#ffffff',
    tooltipBorder: darkMode ? '#475569' : '#e2e8f0',
  };

  const colors = {
    average: '#38bdf8',      // Яркий голубой
    maximum: '#34d399',      // Яркий зеленый
    withWeight: '#a78bfa',   // Яркий фиолетовый
    actual: '#fb7185',       // Яркий красный
    reference: '#94a3b8',    // Серый для линий нормативов
    achievement: '#fbbf24',   // Яркий оранжевый для достижений
    standardLine: '#64748b',  // Более темный серый для линий стандартов
    achievementLine: '#f97316' // Оранжевый для линий достижений
  };

  // Отображаем сообщение, если у пользователя нет данных
  if (noUserData) {
    return (
      <div className={styles.chartContainer} style={{ backgroundColor: theme.backgroundColor }}>
        <h3 style={{ color: theme.textColor }}>{title}</h3>
        <div className={styles.noDataMessage} style={{ color: theme.textColor }}>
          <div className={styles.messageIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
          </div>
          <p>{message || 'Необходимо загрузить данные о ваших тренировках для построения персонального прогноза.'}</p>
          <div className={styles.actionHint}>
            <p>Используйте форму "Добавить тренировку" слева для внесения ваших данных.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.chartContainer} style={{ backgroundColor: theme.backgroundColor }}>
      <h3 style={{ color: theme.textColor }}>{title}</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 10, right: 80, left: 20, bottom: 10 }}
        >
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={theme.gridColor}
            opacity={0.3}
          />
          <XAxis
            dataKey="date"
            stroke={theme.textColor}
            tick={{ fill: theme.textColor, fontSize: 13 }}
            label={{ 
              value: xAxisLabel, 
              fill: theme.textColor, 
              fontSize: 13,
              dy: 15 
            }}
            interval="preserveStartEnd"
            minTickGap={40}
            tickFormatter={(value) => value.split('-').reverse().join('.')}
            padding={{ left: 10, right: 10 }}
          />
          <YAxis
            stroke={theme.textColor}
            tick={{ fill: theme.textColor, fontSize: 13 }}
            label={{ 
              value: yAxisLabel, 
              angle: -90, 
              fill: theme.textColor,
              fontSize: 13,
              dx: -30 
            }}
            padding={{ top: 10, bottom: 10 }}
          />
          <Tooltip 
            content={<CustomTooltip theme={theme} />}
            animationDuration={300}
            cursor={{ stroke: theme.gridColor, strokeWidth: 1 }}
          />
          <Legend 
            wrapperStyle={{ 
              color: theme.textColor,
              paddingTop: 16,
              fontSize: 13
            }}
            verticalAlign="bottom"
            height={36}
            iconSize={10}
            iconType="circle"
          />
          
          {/* Линии прогнозов */}
          <Line
            type="monotone"
            dataKey="average"
            stroke={colors.average}
            strokeWidth={2}
            dot={false}
            name="Прогноз (среднее)"
            connectNulls={true}
            animationDuration={1000}
            animationBegin={0}
          />
          <Line
            type="monotone"
            dataKey="maximum"
            stroke={colors.maximum}
            strokeWidth={2}
            dot={false}
            name="Прогноз (максимум)"
            connectNulls={true}
            animationDuration={1000}
            animationBegin={200}
          />
          <Line
            type="monotone"
            dataKey="withWeight"
            stroke={colors.withWeight}
            strokeWidth={2}
            dot={false}
            name="Прогноз (с весом)"
            connectNulls={true}
            animationDuration={1000}
            animationBegin={400}
          />
          <Line
            type="monotone"
            dataKey="actual"
            stroke={colors.actual}
            strokeWidth={2.5}
            dot={{ 
              stroke: colors.actual, 
              strokeWidth: 2, 
              r: 4,
              fill: theme.backgroundColor
            }}
            activeDot={{
              stroke: colors.actual,
              strokeWidth: 2,
              r: 6,
              fill: theme.backgroundColor
            }}
            name="Фактические данные"
            animationDuration={1000}
            animationBegin={600}
          />
          
          {/* Горизонтальные линии нормативов */}
          {standards && standards.map((standard, index) => (
            <ReferenceLine
              key={`hline-${index}`}
              y={standard.value}
              stroke={colors.standardLine}
              strokeDasharray="3 3"
              strokeWidth={1}
              label={{
                value: standard.rank,
                position: 'right',
                fill: theme.textColor,
                fontSize: 13,
                opacity: 0.9,
                padding: 5
              }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// В конце файла изменим функцию проверки обновления
function arePropsEqual(prevProps, nextProps) {
  // Сравнение timestamp - если они равны, то данные не изменились
  if (prevProps.timestamp === nextProps.timestamp) {
    return true; // Пропускаем обновление, если timestamp не изменился
  }

  // Проверяем только важные изменения, игнорируя незначительные
  if (nextProps.data && nextProps.data.length > 0) {
    // Всегда обновляем при наличии данных и разных timestamp
    return false;
  }
  
  // Для режима без данных пользователя, используем оптимизированное сравнение
  if (nextProps.noUserData) {
    const equalProps = (
      prevProps.noUserData === nextProps.noUserData &&
      prevProps.message === nextProps.message &&
      prevProps.darkMode === nextProps.darkMode
    );
    return equalProps;
  }
  
  // В других случаях по умолчанию обновляем
  return false;
}

// Оборачиваем компонент в React.memo с нашей функцией сравнения
const Chart = React.memo(ChartComponent, arePropsEqual);

export default Chart;