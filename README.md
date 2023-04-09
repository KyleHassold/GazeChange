<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/KyleHassold/GazeChange">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">Gaze Change</h3>

  <p align="center">
    CMSC472 Project
    <br />
    <a href="https://github.com/KyleHassold/GazeChange"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/KyleHassold/GazeChange">View Demo</a>
    ·
    <a href="https://github.com/KyleHassold/GazeChange/issues">Report Bug</a>
    ·
    <a href="https://github.com/KyleHassold/GazeChange/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/KyleHassold/GazeChange)

Video calls have grown in popularity but still lack in certain pieces of functionality that
come naturally to in-person meetings. One such lack is eye contact. During video calls,
users will be looking at the content that appear on their screens despite the camera not
being at that location. The different in positioning between the content on the monitor and
the camera causes a disconnect resulting in each user not making proper eye contact. Our
goal is to utilize artificial intelligence to adjust the eye positions of each user to be
directed at the camera. While this change is small, we believe that the adjustment will
provide a more in-person feel to the virtual meetings that connect our world.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![MPIIFazeGaze][MPIIFazeGaze-shield]][MPIIFazeGaze-url]
* [![PyTorch][PyTorch-shield]][PyTorch-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/KyleHassold/GazeChange.git
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Simple model for detecting gaze target or detecting if gaze is directed at the camera
  - [ ] Utilize full face preferably
  - [ ] Determine if we can crop out black background
- [ ] Simple GAN model for reproducing eyes (No gaze change)
  - [ ] Input section of image containing eyes
  - [ ] Generate that same region to splice into original image
  - [ ] Feed full image into discriminator
- [ ] Add secondary discriminator to GAN to train it to adjust gaze
  - [ ] Potentially utilize pretrained GAN and Gaze Detector
- [ ] Additional Features:
  - [ ] Analyze gaze direction before applying generator

See the [open issues](https://github.com/KyleHassold/GazeChange/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Kyle Hassold - khassold@umd.edu

Khushi Bhansali - bhansali.khushi19@gmail.com

Dotun Oladimejij - ooladim1@terpmail.umd.edu

Anaya Steward - aksteward225@gmail.com

Project Link: [https://github.com/KyleHassold/GazeChange](https://github.com/KyleHassold/GazeChange)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/KyleHassold/GazeChange.svg?style=for-the-badge
[contributors-url]: https://github.com/KyleHassold/GazeChange/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/KyleHassold/GazeChange.svg?style=for-the-badge
[forks-url]: https://github.com/KyleHassold/GazeChange/network/members
[stars-shield]: https://img.shields.io/github/stars/KyleHassold/GazeChange.svg?style=for-the-badge
[stars-url]: https://github.com/KyleHassold/GazeChange/stargazers
[issues-shield]: https://img.shields.io/github/issues/KyleHassold/GazeChange.svg?style=for-the-badge
[issues-url]: https://github.com/KyleHassold/GazeChange/issues
[product-screenshot]: images/screenshot.png
[MPIIFazeGaze-shield]: https://img.shields.io/badge/MPIIFazeGaze-db6410?style=for-the-badge
[MPIIFazeGaze-url]: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/