/**
 * Real-time waveform renderer using Canvas 2D.
 *
 * Draws a scrolling amplitude waveform — mic input in orange,
 * file playback in blue. Uses peak-bucketing for smooth rendering
 * regardless of sample rate vs canvas width.
 *
 * Usage:
 *   const wf = new WaveformRenderer(canvasElement);
 *   wf.start();
 *   wf.pushSamples(float32Array);  // call from audio callback
 *   wf.stop();
 */

export class WaveformRenderer {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {object} opts
     * @param {number} opts.durationSecs - visible window in seconds (default: 3)
     * @param {number} opts.sampleRate - expected sample rate (default: 16000)
     * @param {string} opts.color - waveform stroke color (default: CSS var --orange)
     * @param {string} opts.bgColor - background fill (default: CSS var --bg)
     * @param {string} opts.lineColor - center line color (default: CSS var --border)
     */
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        this.durationSecs = opts.durationSecs || 3;
        this.sampleRate = opts.sampleRate || 16000;
        this.color = opts.color || '#cf6a4c';       // --orange
        this.bgColor = opts.bgColor || '#151515';   // --bg
        this.lineColor = opts.lineColor || '#303030'; // --border

        // Ring buffer: store enough samples for the visible window
        this.capacity = this.durationSecs * this.sampleRate;
        this.buffer = new Float32Array(this.capacity);
        this.writePos = 0;
        this.len = 0;

        this.animFrameId = null;
        this.running = false;

        // Handle high-DPI displays
        this._resizeCanvas();
        this._resizeObserver = new ResizeObserver(() => this._resizeCanvas());
        this._resizeObserver.observe(this.canvas);
    }

    /** Start the render loop. */
    start() {
        if (this.running) return;
        this.running = true;
        this._draw();
    }

    /** Stop rendering. */
    stop() {
        this.running = false;
        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }
    }

    /** Clear the buffer and canvas. */
    clear() {
        this.writePos = 0;
        this.len = 0;
        this.buffer.fill(0);
        this._drawFrame();
    }

    /** Push new audio samples into the ring buffer. */
    pushSamples(samples) {
        const n = samples.length;
        if (n >= this.capacity) {
            // Only keep the tail
            const start = n - this.capacity;
            this.buffer.set(samples.subarray(start));
            this.writePos = 0;
            this.len = this.capacity;
            return;
        }

        const spaceToEnd = this.capacity - this.writePos;
        if (n <= spaceToEnd) {
            this.buffer.set(samples, this.writePos);
        } else {
            this.buffer.set(samples.subarray(0, spaceToEnd), this.writePos);
            this.buffer.set(samples.subarray(spaceToEnd), 0);
        }
        this.writePos = (this.writePos + n) % this.capacity;
        this.len = Math.min(this.len + n, this.capacity);
    }

    /** Set the waveform color (e.g., switch between mic/file). */
    setColor(color) {
        this.color = color;
    }

    /** Destroy the renderer and clean up. */
    destroy() {
        this.stop();
        this._resizeObserver.disconnect();
    }

    // --- Private ---

    _resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this._drawFrame();
    }

    _draw() {
        if (!this.running) return;
        this._drawFrame();
        this.animFrameId = requestAnimationFrame(() => this._draw());
    }

    _drawFrame() {
        const ctx = this.ctx;
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;

        // Background
        ctx.fillStyle = this.bgColor;
        ctx.fillRect(0, 0, w, h);

        // Center line
        const midY = h / 2;
        ctx.strokeStyle = this.lineColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(w, midY);
        ctx.stroke();

        if (this.len === 0) return;

        // Get peaks for each pixel column
        const peaks = this._computePeaks(w);

        // Draw waveform as mirrored bars
        ctx.fillStyle = this.color;
        const barWidth = Math.max(1, w / peaks.length);

        for (let i = 0; i < peaks.length; i++) {
            const amp = peaks[i];
            const barHeight = amp * (h * 0.9); // 90% of canvas height
            const x = i * barWidth;
            ctx.fillRect(x, midY - barHeight / 2, Math.max(1, barWidth - 0.5), barHeight);
        }
    }

    _computePeaks(width) {
        if (width === 0 || this.len === 0) return [];

        const samplesPerBucket = this.len / width;
        const peaks = new Float32Array(width);

        for (let i = 0; i < width; i++) {
            const start = Math.floor(i * samplesPerBucket);
            const end = Math.floor((i + 1) * samplesPerBucket);

            let maxVal = 0;
            for (let j = start; j < end && j < this.len; j++) {
                // Read in chronological order from ring buffer
                const idx = (this.writePos - this.len + j + this.capacity) % this.capacity;
                const v = Math.abs(this.buffer[idx]);
                if (v > maxVal) maxVal = v;
            }
            peaks[i] = maxVal;
        }

        return peaks;
    }
}
