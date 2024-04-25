import React from "react";
import Navbar from "./Navbar";
import MainContent from "./MainContent";
import Footer from "./Footer.js";
import AudioRecorder from "./AudioRecorder.js";
import "./App.css"; // Assuming you have an App.css file for global styles
import image from "./logoiiits.png"

function App() {
  return (
    <div>
      <header>
        <div style={{ display: "flex", alignItems: "center", textAlign: "center", paddingLeft: "100px" }}>
          {/* Example logo image; replace 'logoSrc' with your actual logo path */}
          <img
            src={image}
            alt="IIIT Sricity Logo"
            style={{ marginRight: "8px", height: "50px" }}
          />
          <h1 style={{ textAlign: "center" }}>
            IIIT Sricity's Spoken Keyword Spotting
          </h1>
        </div>
        <Navbar />
      </header>
      <MainContent />
      <AudioRecorder />
      <Footer />
    </div>
  );
}

export default App;
