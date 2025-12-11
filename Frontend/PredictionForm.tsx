import { useState } from 'react';
import styled from 'styled-components';
import { theme } from '../theme';
import { Card, CardTitle } from './ui/Card';
import { Button } from './ui/Button';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';
import { predictSingle } from '../config/api';
import { PredictionResult } from '../utils/types';

const FormGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 24px;
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  
  label {
    font-size: 0.9rem;
    font-weight: 500;
    color: ${theme.colors.darkNavy};
  }
  
  input, select {
    padding: 10px 12px;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    font-family: inherit;
    font-size: 0.95rem;
    transition: border-color 0.2s;
    
    &:focus {
      outline: none;
      border-color: ${theme.colors.primaryStrong};
      box-shadow: 0 0 0 3px rgba(46, 85, 153, 0.1);
    }
  }
`;

const ResultCard = styled(motion.div) <{ $risk: 'Low' | 'Medium' | 'High' }>`
  margin-top: 24px;
  padding: 20px;
  border-radius: 12px;
  background-color: ${props => {
        if (props.$risk === 'High') return '#FEF2F2';
        if (props.$risk === 'Medium') return '#FFFBEB';
        return '#ECFDF5';
    }};
  border: 1px solid ${props => {
        if (props.$risk === 'High') return '#FCA5A5';
        if (props.$risk === 'Medium') return '#FCD34D';
        return '#6EE7B7';
    }};
  
  h4 {
    color: ${props => {
        if (props.$risk === 'High') return '#991B1B';
        if (props.$risk === 'Medium') return '#92400E';
        return '#065F46';
    }};
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  p {
    color: ${theme.colors.text};
    font-size: 0.95rem;
  }

  ul {
    margin-top: 8px;
    padding-left: 20px;
    color: ${theme.colors.text};
    font-size: 0.9rem;
  }
`;

export const PredictionForm = () => {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [formData, setFormData] = useState({
        age: '',
        partialDischarge: '',
        neutralCorrosion: '',
        loading: '',
        visualCondition: 'Good'
    });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setResult(null);

        try {
            const prediction = await predictSingle({
                id: 'SINGLE-PRED',
                age: Number(formData.age),
                partialDischarge: Number(formData.partialDischarge),
                neutralCorrosion: Number(formData.neutralCorrosion),
                loading: Number(formData.loading),
                visualCondition: formData.visualCondition as 'Good' | 'Medium' | 'Poor'
            });

            setResult(prediction);
        } catch (error) {
            console.error("Prediction failed", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <section id="prediction" style={{ padding: '2rem 0' }}>
            <Card>
                <CardTitle>Single Cable Prediction</CardTitle>
                <form onSubmit={handleSubmit}>
                    <FormGrid>
                        <FormGroup>
                            <label>Age (years)</label>
                            <input
                                name="age"
                                type="number"
                                placeholder="e.g. 15"
                                required
                                value={formData.age}
                                onChange={handleChange}
                            />
                        </FormGroup>
                        <FormGroup>
                            <label>Partial Discharge (0.0 - 1.0)</label>
                            <input
                                name="partialDischarge"
                                type="number"
                                step="0.01"
                                min="0"
                                max="1"
                                placeholder="e.g. 0.25"
                                required
                                value={formData.partialDischarge}
                                onChange={handleChange}
                            />
                        </FormGroup>
                        <FormGroup>
                            <label>Neutral Corrosion (0.4 - 1.0)</label>
                            <input
                                name="neutralCorrosion"
                                type="number"
                                step="0.01"
                                min="0"
                                max="1"
                                placeholder="e.g. 0.8"
                                required
                                value={formData.neutralCorrosion}
                                onChange={handleChange}
                            />
                        </FormGroup>
                        <FormGroup>
                            <label>Loading (Amps)</label>
                            <input
                                name="loading"
                                type="number"
                                placeholder="e.g. 450"
                                required
                                value={formData.loading}
                                onChange={handleChange}
                            />
                        </FormGroup>
                        <FormGroup>
                            <label>Visual Condition</label>
                            <select
                                name="visualCondition"
                                required
                                value={formData.visualCondition}
                                onChange={handleChange}
                            >
                                <option value="Good">Good</option>
                                <option value="Medium">Medium</option>
                                <option value="Poor">Poor</option>
                            </select>
                        </FormGroup>
                    </FormGrid>

                    <Button type="submit" disabled={loading}>
                        {loading ? 'Analyzing...' : 'Predict Health Index'}
                    </Button>
                </form>

                <AnimatePresence>
                    {result && (
                        <ResultCard
                            $risk={result.riskLevel}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                        >
                            <h4>
                                {result.riskLevel === 'High' && <AlertCircle size={20} />}
                                {result.riskLevel === 'Medium' && <AlertTriangle size={20} />}
                                {result.riskLevel === 'Low' && <CheckCircle size={20} />}
                                Risk Level: {result.riskLevel}
                            </h4>
                            <p><strong>Predicted Health Index:</strong> {result.predictedHealthIndex}</p>
                            {result.explanation.length > 0 && (
                                <ul>
                                    {result.explanation.map((exp, i) => <li key={i}>{exp}</li>)}
                                </ul>
                            )}
                        </ResultCard>
                    )}
                </AnimatePresence>
            </Card>
        </section>
    );
};
