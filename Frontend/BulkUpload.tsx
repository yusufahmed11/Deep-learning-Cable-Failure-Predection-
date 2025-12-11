import { useState, useMemo } from 'react';
import styled from 'styled-components';
import { theme } from '../theme';
import { Card, CardTitle } from './ui/Card';
import { Button } from './ui/Button';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, Download, Filter, ChevronLeft, ChevronRight, X, Info, CheckCircle } from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
    PieChart, Pie, Legend
} from 'recharts';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { predictBulk } from '../config/api';
import { CableData, PredictionResult } from '../utils/types';
import { mapColumns } from '../utils/logic';

// --- Styled Components ---

const UploadArea = styled.div`
  border: 2px dashed #CBD5E1;
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: #F8FAFC;
  
  &:hover {
    border-color: ${theme.colors.primaryStrong};
    background: #F1F5F9;
  }
`;

const ControlsContainer = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
  gap: 16px;
`;

const FilterGroup = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
`;

const Select = styled.select`
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid #E5E7EB;
  font-family: inherit;
`;

const TableContainer = styled.div`
  overflow-x: auto;
  border-radius: 8px;
  border: 1px solid #E5E7EB;
  max-height: 600px; /* Scrollable */
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  
  th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #E5E7EB;
    white-space: nowrap;
  }
  
  th {
    background-color: #F9FAFB;
    font-weight: 600;
    color: ${theme.colors.textLight};
    cursor: pointer;
    user-select: none;
    position: sticky;
    top: 0;
    z-index: 10;
    
    &:hover {
      background-color: #F3F4F6;
    }
  }
  
  tr:hover {
    background-color: #F9FAFB;
  }
`;

const Pagination = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 16px;
  padding: 0 8px;
`;

const PageButton = styled.button`
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  padding: 6px 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  &:hover:not(:disabled) {
    background: #F3F4F6;
  }
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 32px;
  height: 300px;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    height: auto;
  }
`;

const ChartCard = styled.div`
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 16px;
  height: 300px;
  display: flex;
  flex-direction: column;
`;

const SidePanel = styled(motion.div)`
  position: fixed;
  top: 0;
  right: 0;
  width: 400px;
  height: 100vh;
  background: white;
  box-shadow: -4px 0 15px rgba(0,0,0,0.1);
  z-index: 100;
  padding: 24px;
  overflow-y: auto;
  
  @media (max-width: 480px) {
    width: 100%;
  }
`;

const Overlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0,0,0,0.3);
  z-index: 99;
`;

const RiskBadge = styled.span<{ $risk: string }>`
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
  background-color: ${props => props.$risk === 'High' ? '#FEF2F2' : props.$risk === 'Medium' ? '#FFFBEB' : '#ECFDF5'};
  color: ${props => props.$risk === 'High' ? '#991B1B' : props.$risk === 'Medium' ? '#92400E' : '#065F46'};
`;

const MappingInfo = styled.div`
  background: #EFF6FF;
  border: 1px solid #BFDBFE;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: #1E40AF;
  font-size: 0.9rem;
`;

// --- Main Component ---

export const BulkUpload = () => {
    const [data, setData] = useState<PredictionResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [fileName, setFileName] = useState<string | null>(null);
    const [mappingSuccess, setMappingSuccess] = useState(false);

    // Filtering & Sorting
    const [filterRisk, setFilterRisk] = useState<'All' | 'Low' | 'Medium' | 'High'>('All');
    const [sortConfig, setSortConfig] = useState<{ key: keyof PredictionResult; direction: 'asc' | 'desc' } | null>(null);

    // Pagination
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 20;

    // Side Panel
    const [selectedCable, setSelectedCable] = useState<PredictionResult | null>(null);

    // --- Handlers ---

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setFileName(file.name);
        setLoading(true);
        setError(null);
        setData([]);
        setMappingSuccess(false);

        try {
            // Call API directly with file
            const results = await predictBulk(file);

            setData(results);
            setMappingSuccess(true);
            setCurrentPage(1);

        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    const handleDownload = () => {
        if (data.length === 0) return;

        const csvData = data.map(row => ({
            ID: row.id,
            Age: row.age,
            Partial_Discharge: row.partialDischarge,
            Visual_Condition: row.visualCondition,
            Neutral_Corrosion: row.neutralCorrosion,
            Loading: row.loading,
            Predicted_Health_Index: row.predictedHealthIndex,
            Risk_Level: row.riskLevel
        }));

        const csv = Papa.unparse(csvData);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'prediction_results.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    // --- Derived State ---

    const filteredData = useMemo(() => {
        if (filterRisk === 'All') return data;
        return data.filter(item => item.riskLevel === filterRisk);
    }, [data, filterRisk]);

    const sortedData = useMemo(() => {
        if (!sortConfig) return filteredData;
        return [...filteredData].sort((a, b) => {
            const aVal = a[sortConfig.key];
            const bVal = b[sortConfig.key];
            if (aVal === undefined || bVal === undefined) return 0;

            if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
            if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
    }, [filteredData, sortConfig]);

    const paginatedData = useMemo(() => {
        const start = (currentPage - 1) * itemsPerPage;
        return sortedData.slice(start, start + itemsPerPage);
    }, [sortedData, currentPage]);

    const totalPages = Math.ceil(sortedData.length / itemsPerPage);

    const riskStats = useMemo(() => {
        const counts = { Low: 0, Medium: 0, High: 0 };
        filteredData.forEach(d => counts[d.riskLevel]++);
        return [
            { name: 'Low', value: counts.Low, color: '#4CAF50' },
            { name: 'Medium', value: counts.Medium, color: '#FF9800' },
            { name: 'High', value: counts.High, color: '#E53935' }
        ];
    }, [filteredData]);

    // --- Render ---

    return (
        <section id="bulk" style={{ padding: '2rem 0' }}>
            <Card>
                <CardTitle>Bulk Upload - File Prediction</CardTitle>

                {/* Upload Section */}
                {!data.length && !loading && (
                    <>
                        <input
                            type="file"
                            id="file-upload"
                            style={{ display: 'none' }}
                            accept=".csv,.xlsx,.xls"
                            onChange={handleFileUpload}
                        />
                        <label htmlFor="file-upload">
                            <UploadArea>
                                <Upload size={40} color={theme.colors.textLight} style={{ marginBottom: '16px' }} />
                                <p style={{ fontWeight: 500, color: theme.colors.darkNavy }}>
                                    {fileName || "Click to upload CSV/Excel file"}
                                </p>
                                <p style={{ fontSize: '0.85rem', color: theme.colors.textLight, marginTop: '8px' }}>
                                    Supports .csv and .xlsx. Auto-maps columns (see Data Guide).
                                </p>
                            </UploadArea>
                        </label>
                        {error && (
                            <p style={{ color: theme.colors.danger, marginTop: '16px', textAlign: 'center', fontWeight: 500, padding: '10px', background: '#FEF2F2', borderRadius: '8px' }}>{error}</p>
                        )}
                    </>
                )}

                {loading && (
                    <div style={{ textAlign: 'center', padding: '40px' }}>
                        <p>Processing file and generating predictions...</p>
                    </div>
                )}

                {/* Results Section */}
                {data.length > 0 && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>

                        {mappingSuccess && (
                            <MappingInfo>
                                <CheckCircle size={20} />
                                Mapped Columns → Uploaded → Standardized
                            </MappingInfo>
                        )}

                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px', flexWrap: 'wrap', gap: '10px' }}>
                            <Button $variant="secondary" onClick={() => { setData([]); setFileName(null); setMappingSuccess(false); }}>
                                Upload New File
                            </Button>
                            <Button onClick={handleDownload}>
                                <Download size={16} /> Download CSV
                            </Button>
                        </div>

                        {/* Charts */}
                        <ChartsGrid>
                            <ChartCard>
                                <h4 style={{ marginBottom: '16px', color: theme.colors.darkNavy }}>Risk Distribution</h4>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={riskStats}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                        <XAxis dataKey="name" axisLine={false} tickLine={false} />
                                        <YAxis axisLine={false} tickLine={false} />
                                        <Tooltip />
                                        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                            {riskStats.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </ChartCard>

                            <ChartCard>
                                <h4 style={{ marginBottom: '16px', color: theme.colors.darkNavy }}>Risk Percentage</h4>
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={riskStats}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={80}
                                            paddingAngle={5}
                                            dataKey="value"
                                        >
                                            {riskStats.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Pie>
                                        <Tooltip />
                                        <Legend verticalAlign="bottom" height={36} />
                                    </PieChart>
                                </ResponsiveContainer>
                            </ChartCard>
                        </ChartsGrid>

                        {/* Controls */}
                        <ControlsContainer>
                            <FilterGroup>
                                <Filter size={16} color={theme.colors.textLight} />
                                <span style={{ fontSize: '0.9rem', color: theme.colors.text }}>Filter Risk:</span>
                                <Select
                                    value={filterRisk}
                                    onChange={(e) => { setFilterRisk(e.target.value as any); setCurrentPage(1); }}
                                >
                                    <option value="All">All</option>
                                    <option value="Low">Low</option>
                                    <option value="Medium">Medium</option>
                                    <option value="High">High</option>
                                </Select>
                            </FilterGroup>

                            <span style={{ fontSize: '0.9rem', color: theme.colors.textLight }}>
                                Showing {paginatedData.length} of {filteredData.length} entries
                            </span>
                        </ControlsContainer>

                        {/* Table */}
                        <TableContainer>
                            <Table>
                                <thead>
                                    <tr>
                                        <th onClick={() => setSortConfig({ key: 'id', direction: sortConfig?.direction === 'asc' ? 'desc' : 'asc' })}>ID</th>
                                        <th onClick={() => setSortConfig({ key: 'age', direction: sortConfig?.direction === 'asc' ? 'desc' : 'asc' })}>Age</th>
                                        <th onClick={() => setSortConfig({ key: 'partialDischarge', direction: sortConfig?.direction === 'asc' ? 'desc' : 'asc' })}>PD</th>
                                        <th onClick={() => setSortConfig({ key: 'neutralCorrosion', direction: sortConfig?.direction === 'asc' ? 'desc' : 'asc' })}>Corrosion</th>
                                        <th onClick={() => setSortConfig({ key: 'loading', direction: sortConfig?.direction === 'asc' ? 'desc' : 'asc' })}>Loading</th>
                                        <th onClick={() => setSortConfig({ key: 'predictedHealthIndex', direction: sortConfig?.direction === 'asc' ? 'desc' : 'asc' })}>Health Index</th>
                                        <th onClick={() => setSortConfig({ key: 'riskLevel', direction: sortConfig?.direction === 'asc' ? 'desc' : 'asc' })}>Risk Level</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {paginatedData.map((row) => (
                                        <tr key={row.id}>
                                            <td>{row.id}</td>
                                            <td>{row.age}</td>
                                            <td>{row.partialDischarge}</td>
                                            <td>{row.neutralCorrosion}</td>
                                            <td>{row.loading}</td>
                                            <td>{row.predictedHealthIndex}</td>
                                            <td><RiskBadge $risk={row.riskLevel}>{row.riskLevel}</RiskBadge></td>
                                            <td>
                                                <Button
                                                    $variant="secondary"
                                                    style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                                                    onClick={() => setSelectedCable(row)}
                                                >
                                                    Details
                                                </Button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </Table>
                        </TableContainer>

                        {/* Pagination */}
                        <Pagination>
                            <PageButton
                                disabled={currentPage === 1}
                                onClick={() => setCurrentPage(p => p - 1)}
                            >
                                <ChevronLeft size={16} /> Previous
                            </PageButton>
                            <span style={{ fontSize: '0.9rem', color: theme.colors.text }}>
                                Page {currentPage} of {totalPages}
                            </span>
                            <PageButton
                                disabled={currentPage === totalPages}
                                onClick={() => setCurrentPage(p => p + 1)}
                            >
                                Next <ChevronRight size={16} />
                            </PageButton>
                        </Pagination>

                    </motion.div>
                )}
            </Card>

            {/* Side Panel */}
            <AnimatePresence>
                {selectedCable && (
                    <>
                        <Overlay
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setSelectedCable(null)}
                        />
                        <SidePanel
                            initial={{ x: '100%' }}
                            animate={{ x: 0 }}
                            exit={{ x: '100%' }}
                            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                                <h3 style={{ fontSize: '1.5rem', color: theme.colors.darkNavy }}>Cable Details</h3>
                                <button
                                    onClick={() => setSelectedCable(null)}
                                    style={{ background: 'none', border: 'none', cursor: 'pointer' }}
                                >
                                    <X size={24} color={theme.colors.text} />
                                </button>
                            </div>

                            <div style={{ marginBottom: '24px' }}>
                                <h4 style={{ color: theme.colors.textLight, fontSize: '0.9rem', marginBottom: '4px' }}>Cable ID</h4>
                                <p style={{ fontSize: '1.2rem', fontWeight: 600 }}>{selectedCable.id}</p>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '32px' }}>
                                <div style={{ background: '#F8FAFC', padding: '16px', borderRadius: '8px' }}>
                                    <h4 style={{ color: theme.colors.textLight, fontSize: '0.85rem' }}>Health Index</h4>
                                    <p style={{ fontSize: '1.5rem', fontWeight: 700, color: theme.colors.primaryStrong }}>
                                        {selectedCable.predictedHealthIndex}
                                    </p>
                                </div>
                                <div style={{ background: '#F8FAFC', padding: '16px', borderRadius: '8px' }}>
                                    <h4 style={{ color: theme.colors.textLight, fontSize: '0.85rem' }}>Risk Level</h4>
                                    <RiskBadge $risk={selectedCable.riskLevel} style={{ fontSize: '1rem', marginTop: '4px', display: 'inline-block' }}>
                                        {selectedCable.riskLevel}
                                    </RiskBadge>
                                </div>
                            </div>

                            <h4 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Info size={18} /> Contribution Factors
                            </h4>

                            {selectedCable.explanation.length > 0 ? (
                                <ul style={{ paddingLeft: '20px', color: theme.colors.text, lineHeight: '1.6' }}>
                                    {selectedCable.explanation.map((exp, i) => (
                                        <li key={i}>{exp}</li>
                                    ))}
                                </ul>
                            ) : (
                                <p style={{ color: theme.colors.textLight, fontStyle: 'italic' }}>
                                    No specific risk factors detected.
                                </p>
                            )}

                            <div style={{ marginTop: '40px', paddingTop: '20px', borderTop: '1px solid #E5E7EB' }}>
                                <h4 style={{ marginBottom: '12px' }}>Raw Data</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.9rem' }}>
                                    <div><strong>Age:</strong> {selectedCable.age} years</div>
                                    <div><strong>Loading:</strong> {selectedCable.loading} A</div>
                                    <div><strong>PD:</strong> {selectedCable.partialDischarge}</div>
                                    <div><strong>Corrosion:</strong> {selectedCable.neutralCorrosion}</div>
                                    <div><strong>Visual:</strong> {selectedCable.visualCondition}</div>
                                </div>
                            </div>

                        </SidePanel>
                    </>
                )}
            </AnimatePresence>
        </section>
    );
};
