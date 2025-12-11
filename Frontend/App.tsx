import styled from 'styled-components';
import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { PredictionForm } from './components/PredictionForm';
import { BulkUpload } from './components/BulkUpload';
import { ModelPerformance } from './components/ModelPerformance';
import { DataFormatGuide } from './components/DataFormatGuide';

const MainContainer = styled.main`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
`;

function App() {
    return (
        <>
            <Header />
            <Hero />
            <MainContainer>
                <PredictionForm />
                <BulkUpload />
                <DataFormatGuide />
                <ModelPerformance />
            </MainContainer>

            <footer style={{ textAlign: 'center', padding: '40px 0', color: '#6B7280', fontSize: '0.9rem' }}>
                <p>© 2025 Dar AlKahrbaa. All rights reserved.</p>
            </footer>
        </>
    )
}

export default App
