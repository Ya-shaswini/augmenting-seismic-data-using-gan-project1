'use client';

import { useState, useEffect, useRef } from 'react';
import { Header } from '@/components/layout/Header';
import { RealTimeTab } from '@/components/tabs/RealTimeTab';
import { GANDenoisingTab } from '@/components/tabs/GANDenoisingTab';
import { Activity, Layers } from 'lucide-react';
import { cn } from '@/lib/utils';

// Types
interface Message {
    id: string;
    sender: string;
    content: string;
    type: 'text' | 'waveform' | 'alert';
    timestamp: string;
    data?: number[];
}

export default function Home() {
    const [activeTab, setActiveTab] = useState<'monitoring' | 'denoising'>('monitoring');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);

    // WebSocket Logic
    const connect = () => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;
        wsRef.current = new WebSocket('ws://localhost:8000/ws/stream/user_demo');

        wsRef.current.onopen = () => {
            setIsConnected(true);
            console.log('Connected');
        };

        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);

            // Handle real USGS earthquake events
            if (data.type === 'earthquake_event') {
                addMessage({
                    id: data.event_id || `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    sender: `${data.source} - Real Event`,
                    content: `M${data.magnitude} Earthquake - ${data.location}`,
                    type: 'alert',
                    timestamp: new Date().toLocaleTimeString(),
                    data: undefined // Real events don't have waveform data yet
                });

                console.log('Real earthquake detected:', data);
            }
            // Handle simulated alerts (legacy)
            else if (data.type === 'seismic_alert') {
                addMessage({
                    id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    sender: 'System',
                    content: `ALERT: ${data.severity} severity event at ${data.location}`,
                    type: 'alert',
                    timestamp: new Date().toLocaleTimeString()
                });
            }
            // Handle info messages
            else if (data.type === 'info') {
                console.log('WebSocket info:', data.content);
            }
        };

        wsRef.current.onclose = () => {
            setIsConnected(false);
            wsRef.current = null;
        };
    };

    const disconnect = () => {
        if (wsRef.current) wsRef.current.close();
    };

    const toggleConnection = () => {
        if (isConnected) disconnect();
        else connect();
    }

    useEffect(() => {
        connect();
        return () => disconnect();
    }, []);

    const addMessage = (msg: Message) => {
        setMessages(prev => [...prev, msg]);
    };

    // Simulation for demo purposes if backend isn't pushing data fast enough
    useEffect(() => {
        if (!isConnected) return;
        const interval = setInterval(() => {
            if (document.hidden) return;
            // Simulate random event occasionally
            if (Math.random() > 0.8) {
                const isWaveform = Math.random() > 0.5;
                if (isWaveform) {
                    const dummyData = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
                    addMessage({
                        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                        sender: 'Sensor Simulator',
                        content: 'Seismic Waveform Captured',
                        type: 'waveform',
                        data: dummyData,
                        timestamp: new Date().toLocaleTimeString()
                    });
                }
            }
        }, 5000);
        return () => clearInterval(interval);
    }, [isConnected]);


    const handleDenoise = async (originalData: number[]) => {
        console.log("Switching to Denoise tab with data", originalData);
        setActiveTab('denoising');
    };

    const handleDownload = (data: number[]) => {
        const csvContent = "data:text/csv;charset=utf-8," + data.map((e, i) => `${i},${e}`).join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `seismic_data_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <main className="flex h-screen bg-background overflow-hidden font-sans text-foreground">
            <div className="flex-1 flex flex-col h-full relative">
                <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 -z-10" />

                <Header
                    isConnected={isConnected}
                    toggleConnection={toggleConnection}
                />

                {/* Tab Navigation */}
                <div className="px-6 pt-4 border-b border-white/5 flex gap-6 text-sm font-medium overflow-x-auto">
                    <button
                        onClick={() => setActiveTab('monitoring')}
                        className={cn("pb-3 border-b-2 transition-colors flex items-center gap-2 whitespace-nowrap",
                            activeTab === 'monitoring' ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-white")}
                    >
                        <Activity size={16} /> Real-Time Monitoring
                    </button>
                    <button
                        onClick={() => setActiveTab('denoising')}
                        className={cn("pb-3 border-b-2 transition-colors flex items-center gap-2 whitespace-nowrap",
                            activeTab === 'denoising' ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-white")}
                    >
                        <Activity size={16} /> GAN Denoising
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-white/10">
                    {activeTab === 'monitoring' && (
                        <RealTimeTab
                            messages={messages}
                            onDenoise={handleDenoise}
                            onDownload={handleDownload}
                        />
                    )}
                    {activeTab === 'denoising' && <GANDenoisingTab />}
                </div>
            </div>
        </main>
    );
}
