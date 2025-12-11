import styled from 'styled-components';
import { theme } from '../theme';
import { Card, CardTitle } from './ui/Card';
import { motion } from 'framer-motion';

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const StatCard = styled(motion.div)`
  background: white;
  padding: 20px;
  border-radius: 12px;
  border: 1px solid #E5E7EB;
  text-align: center;
  
  h3 {
    font-size: 2rem;
    color: ${theme.colors.primaryStrong};
    margin-bottom: 4px;
  }
  
  p {
    color: ${theme.colors.textLight};
    font-size: 0.9rem;
    font-weight: 500;
  }
`;

const PlotsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const PlotCard = styled.div`
  background: white;
  padding: 16px;
  border-radius: 12px;
  border: 1px solid #E5E7EB;
  
  h4 {
    margin-bottom: 12px;
    color: ${theme.colors.darkNavy};
  }
  
  img {
    width: 100%;
    height: auto;
    border-radius: 8px;
  }
`;

export const ModelPerformance = () => {
    const stats = [
        { label: 'Accuracy', value: '97.40%' },
        { label: 'Precision', value: '0.974' },
        { label: 'Recall', value: '0.974' },
        { label: 'F1-Score', value: '0.974' },
    ];

    return (
        <section id="performance" style={{ padding: '2rem 0', paddingBottom: '4rem' }}>
            <CardTitle style={{ marginBottom: '24px' }}>Model Performance</CardTitle>

            <StatsGrid>
                {stats.map((stat, index) => (
                    <StatCard
                        key={stat.label}
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        viewport={{ once: true }}
                    >
                        <h3>{stat.value}</h3>
                        <p>{stat.label}</p>
                    </StatCard>
                ))}
            </StatsGrid>

            <PlotsGrid>
                <PlotCard>
                    <h4>Training Curves</h4>
                    <img src="/assets/training_curves.png" alt="Training Curves" />
                </PlotCard>
                <PlotCard>
                    <h4>Confusion Matrix</h4>
                    <img src="/assets/confusion_matrix.png" alt="Confusion Matrix" />
                </PlotCard>
            </PlotsGrid>
        </section>
    );
};
