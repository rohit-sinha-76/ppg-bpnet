/**
 * =================================================================
 * FILE: 		app.js
 * PROJECT: 	PPG-BPNet - Advanced Blood Pressure Monitoring
 * VERSION: 	4.0.0
 * DESCRIPTION: Manages all interactivity for the advanced UI.
 * =================================================================
 */

document.addEventListener('DOMContentLoaded', () => {

    const App = {
        // --- Properties ---
        socket: null,
        bpChart: null,
        bpData: { timestamps: [], systolic: [], diastolic: [] },
        isMobileMenuOpen: false,

        /**
         * Initializes all application modules.
         */
        init() {
            gsap.registerPlugin(ScrollTrigger);
            this.initNavigation();
            this.initTabs();
            this.initFileUpload();
            this.initSocketIO();
            this.initAnimations();
            this.initVisualizations();
        },

        // --- Core UI & Navigation ---

        /**
         * Initializes navigation, including smooth scrolling, active state updates,
         * and the mobile menu toggle.
         */
        initNavigation() {
            const navbar = document.getElementById('navbar');
            const navLinks = document.querySelectorAll('.nav-link');
            const mobileMenuBtn = document.getElementById('mobile-menu-btn');
            const navMenu = document.getElementById('nav-menu');

            // Navbar scroll effect
            window.addEventListener('scroll', () => {
                navbar.classList.toggle('scrolled', window.scrollY > 50);
            });

            // Intersection Observer for active nav link
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
                        const sectionId = entry.target.id;
                        navLinks.forEach(link => {
                            link.classList.toggle('active', link.dataset.section === sectionId);
                        });
                    }
                });
            }, { threshold: 0.5 });
            document.querySelectorAll('.section').forEach(section => observer.observe(section));

            // Mobile menu toggle
            mobileMenuBtn.addEventListener('click', () => {
                this.isMobileMenuOpen = !this.isMobileMenuOpen;
                mobileMenuBtn.classList.toggle('active', this.isMobileMenuOpen);
                navMenu.classList.toggle('active', this.isMobileMenuOpen);
            });
            
            // Smooth scroll for all links and close mobile menu
            navLinks.forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    document.querySelector(link.getAttribute('href')).scrollIntoView();
                    if (this.isMobileMenuOpen) {
                        this.isMobileMenuOpen = false;
                        mobileMenuBtn.classList.remove('active');
                        navMenu.classList.remove('active');
                    }
                });
            });
        },

        /**
         * Initializes the tab functionality for the interface section.
         */
        initTabs() {
            const tabContainer = document.querySelector('.tab-navigation');
            if (!tabContainer) return;

            tabContainer.addEventListener('click', (e) => {
                const clickedTab = e.target.closest('.tab-btn');
                if (!clickedTab) return;

                // Update buttons
                tabContainer.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                clickedTab.classList.add('active');

                // Update content panes
                const targetPaneId = `tab-${clickedTab.dataset.tab}`;
                document.querySelectorAll('.tab-pane').forEach(pane => {
                    pane.classList.toggle('active', pane.id === targetPaneId);
                });
            });
        },

        /**
         * Initializes the file upload area with drag & drop functionality.
         */
        initFileUpload() {
            const uploadZone = document.getElementById('upload-zone');
            const fileInput = document.getElementById('file-input');
            const analyzeBtn = document.getElementById('analyze-btn');
            const uploadText = uploadZone.querySelector('h3');

            if (!uploadZone) return;

            const updateUIForFile = (fileName) => {
                uploadText.textContent = fileName;
                uploadZone.classList.add('has-file');
                analyzeBtn.disabled = false;
            };

            uploadZone.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', () => {
                if (fileInput.files.length > 0) {
                    updateUIForFile(fileInput.files[0].name);
                }
            });

            ['dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (eventName === 'dragover') uploadZone.classList.add('dragover');
                    else uploadZone.classList.remove('dragover');
                    
                    if (eventName === 'drop') {
                        fileInput.files = e.dataTransfer.files;
                        if (fileInput.files.length > 0) {
                            updateUIForFile(fileInput.files[0].name);
                        }
                    }
                });
            });
        },

        // --- Real-time & Socket.IO ---

        /**
         * Initializes Socket.IO connection and event handlers.
         */
        initSocketIO() {
            try {
                this.socket = io();
                this.socket.on('connect', () => console.log('Socket.IO: Connected'));
                this.socket.on('new_bp_data', (data) => this.handleNewBPData(data));

                document.getElementById('start-monitoring').addEventListener('click', () => {
                    this.socket.emit('start_monitoring');
                    this.updateMonitoringStatus(true);
                });
                
                document.getElementById('stop-monitoring').addEventListener('click', () => {
                    this.socket.emit('stop_monitoring');
                    this.updateMonitoringStatus(false);
                });

            } catch (e) {
                console.error("Socket.IO failed to initialize.", e);
            }
        },

        /**
         * Updates the UI to reflect the current monitoring status.
         * @param {boolean} isOnline - True if monitoring is active.
         */
        updateMonitoringStatus(isOnline) {
            const statusIndicator = document.querySelector('#monitoring-status .status-indicator');
            const statusText = statusIndicator.querySelector('span');
            
            document.getElementById('start-monitoring').disabled = isOnline;
            document.getElementById('stop-monitoring').disabled = !isOnline;
            
            statusIndicator.classList.toggle('online', isOnline);
            statusIndicator.classList.toggle('offline', !isOnline);
            statusText.textContent = isOnline ? 'Device Online' : 'Device Offline';

            if (isOnline) this.resetAndShowResults();
        },

        /**
         * Handles new BP data from the server.
         * @param {object} data - {sbp, dbp, timestamp}
         */
        handleNewBPData(data) {
            document.getElementById('sbp-value').textContent = data.sbp;
            document.getElementById('dbp-value').textContent = data.dbp;
            this.updateBPStatus(data.sbp, data.dbp);
            this.updateBPChart(data);
            
            const timestampEl = document.getElementById('results-timestamp');
            timestampEl.textContent = `Last updated: ${new Date(data.timestamp * 1000).toLocaleTimeString()}`;
        },

        /**
         * Updates the BP status text and color.
         * @param {number} sbp - Systolic value.
         * @param {number} dbp - Diastolic value.
         */
        updateBPStatus(sbp, dbp) {
            const statusEl = document.getElementById('bp-status');
            let status = '--', colorClass = '';
            if (sbp < 120 && dbp < 80) { status = 'Normal'; colorClass = 'status-normal'; }
            else if (sbp < 130 && dbp < 80) { status = 'Elevated'; colorClass = 'status-elevated'; }
            else if (sbp < 140 || dbp < 90) { status = 'Stage 1 HTN'; colorClass = 'status-stage1'; }
            else { status = 'Stage 2 HTN'; colorClass = 'status-stage2'; }
            
            statusEl.textContent = status;
            statusEl.parentElement.className = `bp-card status ${colorClass}`;
        },
        
        /**
         * Resets the results section and makes it visible.
         */
        resetAndShowResults() {
            const resultsSection = document.getElementById('results-section');
            resultsSection.classList.remove('hidden');
            gsap.fromTo(resultsSection, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5 });
            this.resetBPChart();
        },

        // --- Animations ---

        /**
         * Initializes GSAP animations for the page.
         */
        initAnimations() {
            // Animate sections on scroll
            document.querySelectorAll('.section').forEach(section => {
                gsap.from(section.querySelector('.container'), {
                    scrollTrigger: {
                        trigger: section,
                        start: 'top 80%',
                        toggleActions: 'play none none none',
                    },
                    opacity: 0,
                    y: 50,
                    duration: 1,
                    ease: 'power3.out',
                });
            });

            // Hero content animation
            gsap.from('.hero-content > *', {
                opacity: 0,
                y: 30,
                duration: 0.8,
                stagger: 0.2,
                ease: 'power3.out',
                delay: 0.5
            });
            gsap.from('.hero-visual', {
                opacity: 0,
                scale: 0.8,
                duration: 1,
                ease: 'power3.out',
                delay: 1
            });
        },

        // --- Plotly Visualizations ---

        /**
         * Initializes all static and dynamic charts.
         */
        initVisualizations() {
            const commonLayout = {
                font: { family: 'Inter, sans-serif', color: 'var(--text-light)' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { gridcolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.2)', zerolinecolor: 'rgba(255,255,255,0.2)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.2)', zerolinecolor: 'rgba(255,255,255,0.2)' },
            };

            // MOCK DATA: Generating demo PPG signal for hero section
            const x_hero = Array.from({length: 200}, (_, i) => i);
            const y_hero = x_hero.map(t => Math.sin(t / 10) * Math.sin(t / 5) + Math.random() * 0.1);
            Plotly.newPlot('hero-ppg-chart', [{ x: x_hero, y: y_hero, type: 'scatter', mode: 'lines', line: { color: 'var(--primary-color)', width: 2 } }], 
                { ...commonLayout, showlegend: false, margin: {t:0,b:0,l:0,r:0}, xaxis: {visible: false}, yaxis: {visible: false} }, 
                { responsive: true, staticPlot: true }
            );

            // Other charts would be initialized here...
            // For brevity, only the live BP chart logic is fully implemented below.
        },
        
        /**
         * Resets and initializes the main BP trend chart.
         */
        resetBPChart() {
            this.bpData = { timestamps: [], systolic: [], diastolic: [] };
            if (this.bpChart) Plotly.purge('bp-chart');
            
            const layout = {
                xaxis: { title: 'Time', color: 'var(--text-light)' },
                yaxis: { title: 'BP (mmHg)', range: [50, 180], color: 'var(--text-light)' },
                margin: { t: 20, l: 60, r: 20, b: 50 },
                legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', font: {color: 'var(--text-light)'} },
                font: { family: 'Inter, sans-serif' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                grid: {
                    color: 'rgba(255,255,255,0.1)'
                }
            };

            // MOCK DATA: Empty placeholder chart before real data arrives
            this.bpChart = Plotly.newPlot('bp-chart', [
                { x: [], y: [], type: 'scatter', mode: 'lines+markers', name: 'Systolic', line: { color: 'var(--warning-color)', width: 2 } },
                { x: [], y: [], type: 'scatter', mode: 'lines+markers', name: 'Diastolic', line: { color: 'var(--accent-color)', width: 2 } }
            ], layout, { responsive: true });
        },
        
        /**
         * Updates the live BP chart with new data.
         * @param {object} data - {sbp, dbp, timestamp}
         */
        updateBPChart(data) {
            if (!this.bpChart) this.resetBPChart();

            this.bpData.timestamps.push(new Date(data.timestamp * 1000).toLocaleTimeString());
            this.bpData.systolic.push(data.sbp);
            this.bpData.diastolic.push(data.dbp);

            const maxPoints = 30;
            if (this.bpData.timestamps.length > maxPoints) {
                this.bpData.timestamps.shift();
                this.bpData.systolic.shift();
                this.bpData.diastolic.shift();
            }

            Plotly.update('bp-chart', {
                x: [this.bpData.timestamps, this.bpData.timestamps],
                y: [this.bpData.systolic, this.bpData.diastolic]
            });
        },
    };

    App.init();
});
