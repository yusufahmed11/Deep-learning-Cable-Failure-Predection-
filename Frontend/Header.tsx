import styled from 'styled-components';
import { theme } from '../theme';
import { motion } from 'framer-motion';

const HeaderContainer = styled(motion.header)`
  position: sticky;
  top: 0;
  z-index: 50;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(231, 240, 250, 0.5);
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const LogoSection = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  
  img {
    height: 40px;
    width: auto;
  }
  
  span {
    font-weight: 700;
    font-size: 1.25rem;
    color: ${theme.colors.darkNavy};
  }
`;

const NavLinks = styled.nav`
  display: flex;
  gap: 2rem;
  
  a {
    color: ${theme.colors.text};
    font-weight: 500;
    font-size: 0.95rem;
    transition: color 0.2s;
    cursor: pointer;
    
    &:hover {
      color: ${theme.colors.primaryStrong};
    }
  }

  @media (max-width: 768px) {
    display: none;
  }
`;

export const Header = () => {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <HeaderContainer
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <LogoSection>
        <img src="/assets/logo.png" alt="Dar AlKahrbaa Logo" />
        {/* <span>Dar AlKahrbaa</span> */}
      </LogoSection>
      <NavLinks>
        <a onClick={() => scrollToSection('home')}>Home</a>
        <a onClick={() => scrollToSection('prediction')}>Single Prediction</a>
        <a onClick={() => scrollToSection('bulk')}>Bulk Upload</a>
        <a onClick={() => scrollToSection('guide')}>Data Guide</a>
        <a onClick={() => scrollToSection('performance')}>Model Performance</a>
      </NavLinks>
    </HeaderContainer>
  );
};
