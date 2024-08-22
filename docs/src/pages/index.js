import React from 'react';
import Layout from '@theme/Layout';
import MainTree from '@site/static/img/tree-3.svg';
import Rocket from "@site/static/img/rocket-launch.png"
import SecondTree from '@site/static/img/tree-2.svg';
import Axe from '@site/static/img/axe.svg';
import Logo from '@site/static/img/logo.png';


export default function Home() {
  return (
    <Layout title="Home" description="Home Page">
      <section class="fg-main-desktop">
        <div class="fg-hero-title" style={{ justifyContent: "center", display: "flex", backgroundColor: "white" }}>
          <div class="fg-hero-title-left">
            <MainTree class="margin-vert--xl fg-title-tree margin-horiz--md" />
          </div>
          <div class="fg-hero-title-right">
            <h1 class="text--bold fg-title-text margin-horiz--lg margin-bottom--sm"> Forest <span
              style={{ color: "#50806C" }}>
              Guard</span>
            </h1>
            <h2 class="text--bold margin-horiz--lg" style={{ color: "#232020" }}>
              Tracking desktop app
            </h2>
            <button onClick={() => location.href="./docs/forest-guard-cli/installation"} type="button" class="fg-button button button--lg margin-horiz--lg"  style={{
              alignItems: "center",
              display: "flex",
            }}>
              <img src={Rocket} width="20px" height="20px" class="margin-right--sm" />
              Get started
            </button>
          </div>
        </div>
        <div style={{ backgroundColor: "#F9F9F9" }}>
          <div class="container">
            <div class="row" style={{ justifyContent: "center", display: "flex" }}>
              <div class="col col--4 fg-text-hero margin-horiz--md margin-vert--lg padding-horiz--lg padding-vert--lg shadow--tl" style={{
                textAlign: "center",
              }}>
                <Axe style={{ width: "200px", height: "auto" }} />
                <h2 class="text--bold margin-top--md" style={{ color: "black" }}>
                  Since 2012, <span style={{ color: "#BC4749" }}>illegal logging</span> of thousands of square meters of
                  forest has continued in Russian <span style={{ color: "#6C9A8B" }}>Siberia</span> during the winter
                </h2>
              </div>
              <div class="col col--4 fg-text-hero margin-horiz--md margin-vert--lg padding-horiz--lg padding-vert--lg shadow--tl" style={{
                textAlign: "center",
              }}>
                <SecondTree style={{ width: "200px", height: "auto" }} />
                <h2 class="text--bold margin-top--md" style={{ color: "black" }}>
                  Our mission is to help people recognize and track <span style={{ color: "#6A994E" }}>deforestation</span> over time.
                </h2>
              </div>
            </div>
          </div>
        </div>
        <div style={{ textAlign: "center" }} class="margin-vert--xl">
          <img src={Logo} width="100px" height="100px" />
          <h1 class="text--bold" style={{ fontSize: "50px", marginBottom: "0" }}>Forest Guard</h1>
          <h1 class="text--normal" >Want to learn more?</h1>
          <div style={{
            display: "flex",
            justifyContent: "center",
          }}>
            <button onClick={() => location.href="./blog"} type="button" class="fg-button button button--lg margin-horiz--sm" style={{
              backgroundColor: "white",
              color: "#50806C",
              border: "solid",
              outlineColor: "#50806C",
              outlineWidth: "5px",
            }}>About</button>
            <button onClick={() => location.href="./docs/forest-guard-cli/installation"} type="button" class="fg-button button button--lg margin-horiz--sm" style={{
              backgroundColor: "#50806C",
            }}>Documentation</button>
          </div>
        </div>
      </section>
      <section class="fg-main-mobile">
        <div class="fg-hero-title" style={{ backgroundColor: "white" }}>
          <div class="fg-hero-title-left">
            <MainTree class="margin-vert--sm margin-top--lg" width="130px" height="auto" style={{
              display: "block", marginLeft: "auto", marginRight: "auto"
            }} />
          </div>
          <div class="fg-hero-title-right" style={{
            textAlign: "center",
          }}>
            <h1 class="text--bold margin-bottom--xs" style={{ fontSize: "50px" }}> Forest <span
              style={{ color: "#50806C" }}>
              Guard</span>
            </h1>
            <h2 class="text--bold margin-horiz--lg" style={{ color: "#232020" }}>
              Tracking desktop app
            </h2>
            <div style={{
              display: "flex",
              justifyContent: "center",
            }} class="margin-bottom--lg">
              <button onClick={() => location.href = "./docs/forest-guard-cli/installation"} type="button" class="fg-button button button--lg margin-horiz--lg" style={{
                alignItems: "center",
                display: "flex",
              }}>
                <img src={Rocket} width="20px" height="20px" class="margin-right--sm" />
                Get started
              </button>
            </div>
          </div>
        </div>
        <div style={{ backgroundColor: "#F9F9F9" }}>
          <div class="container padding-vert--lg">
            <div class="fg-text-hero margin-horiz--md padding-horiz--lg padding-top--lg padding-bottom--md shadow--tl" style={{
              textAlign: "center",
            }}>
              <Axe style={{ width: "100px", height: "auto" }} />
              <h3 class="text--bold margin-top--md" style={{ color: "black" }}>
                Since 2012, <span style={{ color: "#BC4749" }}>illegal logging</span> of thousands of square meters of
                forest has continued in Russian <span style={{ color: "#6C9A8B" }}>Siberia</span> during the winter
              </h3>
            </div>
            <div class="fg-text-hero margin-horiz--md margin-top--lg padding-horiz--lg padding-top--lg padding-bottom--md shadow--tl" style={{
              textAlign: "center",
            }}>
              <SecondTree style={{ width: "100px", height: "auto" }} />
              <h3 class="text--bold margin-top--md" style={{ color: "black" }}>
                Our mission is to help people recognize and track <span style={{ color: "#6A994E" }}>deforestation</span> over time.
              </h3>
            </div>
          </div>
        </div>
        <div style={{ textAlign: "center" }} class="margin-vert--lg">
          <img src={Logo} width="100px" height="100px" />
          <h1 class="text--bold" style={{ fontSize: "40px", marginBottom: "0" }}>Forest Guard</h1>
          <h1 class="text--normal" >Want to learn more?</h1>
          <div style={{
            // display: "flex",
            // justifyContent: "center",
          }} class="padding-horiz--lg">
            <button onClick={() => location.href = "./docs/forest-guard-cli/installation"} type="button" class="fg-button button button--lg margin-bottom--sm" style={{
              backgroundColor: "#50806C",
              display: "block",
              margin: "auto",
            }}>Documentation</button>
            <button onClick={() => location.href = "./blog"} type="button" class="fg-button button button--lg" style={{
              backgroundColor: "white",
              color: "#50806C",
              border: "solid",
              outlineColor: "#50806C",
              outlineWidth: "5px",
              display: "block",
              margin: "auto",
            }}>About</button>
          </div>
        </div>
      </section>
    </Layout>
  );
}