import styled from 'styled-components';
import { theme } from '../theme';
import { motion } from 'framer-motion';
import { Button } from './ui/Button';
import { Activity } from 'lucide-react';

const HeroSection = styled.section`
  min-height: 80vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 4rem 2rem;
  background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.8) 0%, rgba(231,240,250,0) 100%);
`;

const Title = styled(motion.h1)`
  font-size: 3.5rem;
  font-weight: 800;
  color: ${theme.colors.darkNavy};
  margin-bottom: 1rem;
  letter-spacing: -0.02em;
  
  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const Subtitle = styled(motion.p)`
  font-size: 1.25rem;
  color: ${theme.colors.textLight};
  max-width: 600px;
  margin-bottom: 0.5rem;
  line-height: 1.6;
`;

const ArabicTagline = styled(motion.p)`
  font-family: 'Poppins', sans-serif; /* Ideally an Arabic font, but Poppins handles it okay-ish or falls back */
  font-size: 1rem;
  color: ${theme.colors.primaryStrong};
  margin-bottom: 2.5rem;
  opacity: 0.9;
`;

export const Hero = () => {
    const scrollToPrediction = () => {
        document.getElementById('prediction')?.scrollIntoView({ behavior: 'smooth' });
    };

    return (
        <HeroSection id="home">
            <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5 }}
            >
                <Activity size={48} color={theme.colors.primaryStrong} style={{ marginBottom: '1rem' }} />
            </motion.div>

            <Title
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.6 }}
            >
                Cable Health Monitor
            </Title>

            <Subtitle
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.6 }}
            >
                Predict cable failure risk for Dar AlKahrbaa networks using advanced deep learning.
            </Subtitle>

            <ArabicTagline
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.6 }}
            >
                نظام ذكي لمتابعة حالة الكابلات والتنبؤ بالأعطال قبل حدوثها
            </ArabicTagline>

            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.6 }}
            >
                <Button onClick={scrollToPrediction}>
                    Start Prediction
                </Button>
            </motion.div>
        </HeroSection>
    );
};
