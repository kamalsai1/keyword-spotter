// Waveform.js
import React, { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';

const WaveformDisplay = ({ audioUrl }) => {
  const waveformRef = useRef(null);

  useEffect(() => {
    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: 'violet',
      progressColor: 'purple',
      height: 100,
    });

    wavesurfer.load(audioUrl);

    return () => wavesurfer.destroy();
  }, [audioUrl]);

  return <div ref={waveformRef} />;
};

export default WaveformDisplay;
