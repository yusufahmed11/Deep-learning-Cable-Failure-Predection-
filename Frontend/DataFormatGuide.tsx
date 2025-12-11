import styled from 'styled-components';
import { theme } from '../theme';
import { Card, CardTitle } from './ui/Card';
import { Button } from './ui/Button';
import { Download } from 'lucide-react';
import { COLUMN_ALIASES, REQUIRED_COLUMNS } from '../utils/types';
import Papa from 'papaparse';

const GuideContainer = styled.section`
  padding: 2rem 0;
`;

const TableWrapper = styled.div`
  overflow-x: auto;
  margin-top: 1.5rem;
  border-radius: 8px;
  border: 1px solid #E5E7EB;
`;

const ExampleTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  
  th, td {
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid #E5E7EB;
  }
  
  th {
    background-color: #F9FAFB;
    font-weight: 600;
    color: ${theme.colors.darkNavy};
  }
  
  tr:last-child td {
    border-bottom: none;
  }
`;

const AliasList = styled.ul`
  list-style: none;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
  margin-top: 16px;
  
  li {
    background: #F8FAFC;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid #E5E7EB;
    
    strong {
      color: ${theme.colors.primaryStrong};
      display: block;
      margin-bottom: 4px;
    }
    
    span {
      color: ${theme.colors.textLight};
      font-size: 0.9rem;
    }
  }
`;

export const DataFormatGuide = () => {
    const handleDownloadTemplate = () => {
        const csv = Papa.unparse([
            {
                ID: 'CBL-001',
                Age: 15,
                Partial_Discharge: 0.25,
                Neutral_Corrosion: 0.45,
                Loading: 450,
                Visual_Condition: 'Good'
            }
        ]);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'cable_data_template.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <GuideContainer id="guide">
            <Card>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                    <CardTitle style={{ marginBottom: 0 }}>Data Format Guide</CardTitle>
                    <Button onClick={handleDownloadTemplate} $variant="secondary">
                        <Download size={16} /> Download CSV Template
                    </Button>
                </div>

                <p style={{ color: theme.colors.text, marginTop: '1rem', marginBottom: '1rem' }}>
                    To ensure accurate predictions, your uploaded CSV or Excel file must follow the specific format below.
                    The system will automatically map columns if they match any of the accepted aliases.
                </p>

                <h4 style={{ color: theme.colors.primaryStrong, marginTop: '1.5rem' }}>Accepted Column Names & Aliases</h4>
                <AliasList>
                    {Object.entries(COLUMN_ALIASES).map(([key, aliases]) => (
                        <li key={key}>
                            <strong>{key}</strong>
                            <span>{aliases.join(', ')}</span>
                        </li>
                    ))}
                </AliasList>

                <h4 style={{ color: theme.colors.primaryStrong, marginTop: '2rem' }}>Example Valid Data</h4>
                <TableWrapper>
                    <ExampleTable>
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Age</th>
                                <th>Partial_Discharge</th>
                                <th>Neutral_Corrosion</th>
                                <th>Loading</th>
                                <th>Visual_Condition</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>CBL-001</td>
                                <td>15</td>
                                <td>0.25</td>
                                <td>0.45</td>
                                <td>450</td>
                                <td>Good</td>
                            </tr>
                            <tr>
                                <td>CBL-002</td>
                                <td>22</td>
                                <td>0.65</td>
                                <td>0.85</td>
                                <td>620</td>
                                <td>Poor</td>
                            </tr>
                            <tr>
                                <td>CBL-003</td>
                                <td>8</td>
                                <td>0.10</td>
                                <td>0.10</td>
                                <td>300</td>
                                <td>Good</td>
                            </tr>
                        </tbody>
                    </ExampleTable>
                </TableWrapper>
            </Card>
        </GuideContainer>
    );
};
