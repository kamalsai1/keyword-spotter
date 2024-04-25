import React from "react";
import { FaHome, FaLanguage, FaEnvelope } from "react-icons/fa"; // Example using react-icons
import "./Navbar.css"
function Navbar() {
  return (
    <nav className="container-fluid navbar">
      <ul>
        <li>
          <a href="#">
            <FaHome />
            Home
          </a>
        </li>
        <li>
          <a href="#">
            <FaLanguage />
            User Language
          </a>
        </li>
        <li>
          <a href="#">
            <FaEnvelope />
            Contact Us
          </a>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;
