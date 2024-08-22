import React from 'react';
import Layout from '@theme/Layout';
import MainTree from '@site/static/img/tree-1-second.svg';
import Logo from '@site/static/img/logo.png';
import Win from '@site/static/img/windows.svg';
import Mac from '@site/static/img/macos.svg';
import Ubuntu from '@site/static/img/ubuntu.svg';


export default function Home() {
  return (
    <Layout title="Download" description="Download Page">
      <section class="fg-main-desktop">
        <div style={{
          textAlign: "center",
          backgroundClip: "white",
        }}>
          <MainTree class="margin-top--lg" height="350px" width="auto" />
          <h1 class="text--bold fg-title-text margin-bottom--lg"> Download </h1>
        </div>
        <div style={{
          backgroundColor: "#F9F9F9",
          display: "flex",
          justifyContent: "center",        
        }} class="">
          <div style={{
            backgroundColor: "white",
            textAlign: "center",
            justifyContent: "center",
          }} class="col col--3 fg-text-hero margin-horiz--md margin-vert--lg shadow--tl padding-horiz--md">
            <h1 class="margin-vert--md margin-bottom--lg" style={{color: "#232020", fontSize: "40px"}}>GUI Client</h1>
            <button onClick={() => location.href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"} type="button" class="fg-button button button--lg margin-bottom--sm padding-horiz--xl" style={{
              display: "block",
              margin: "auto",
              color: "white",
              fontSize: "25px",
              borderRadius: "20px",
            }}>
              Windows
            </button>
            <button onClick={() => location.href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"} type="button" class="fg-button button button--lg margin-bottom--md padding-horiz--xl" style={{
              display: "block",
              margin: "auto",
              color: "white",
              fontSize: "25px",
              borderRadius: "20px",
            }}>
              MacOS
            </button>
            <h5 class="" style={{color: "#232020"}}>bro wanted exe file lmao 💀💀💀</h5>
            <div style={{
              display: "flex",
              justifyContent: "center",
            }} class="margin-vert--lg">
              <Win class="margin-horiz--xs" height="50px" width="auto" />
              <Mac class="margin-horiz--xs" height="50px" width="auto" />
            </div>
          </div>
          <div style={{
            backgroundColor: "white",
            textAlign: "center",
            justifyContent: "center",
          }} class="col col--3 fg-text-hero margin-horiz--md margin-vert--lg shadow--tl padding-horiz--lg">
            <h1 class="margin-vert--md" style={{color: "#232020", fontSize: "40px"}}>Python Package (CLI)</h1>
            <h2 class="margin-vert--sm" style={{color: "#232020"}}>Run in your terminal</h2>
            <button onClick={() => location.href="../docs/forest-guard-cli/installation"} type="button" class="fg-button button button--lg" style={{
              display: "block",
              margin: "auto",
              color: "white",
              fontSize: "25px",
            }}>
                $ pip install fguard
            </button>
            <h3 class="margin-vert--sm" style={{color: "#232020"}}>Beta version</h3>
            <h2 class="margin-vert--sm" style={{color: "#232020"}}>It is recommended to 
            use <span style={{color: "#6A994E"}}>local python virtual environment</span></h2>
            <div style={{
              display: "flex",
              justifyContent: "center",
            }} class="margin-vert--lg">
              <Win class="margin-horiz--xs" height="50px" width="auto" />
              <Mac class="margin-horiz--xs" height="50px" width="auto" />
              <Ubuntu class="margin-horiz--xs" height="50px" width="auto" />
            </div>
          </div>
        </div>
        <div style={{textAlign: "center"}} class="margin-vert--xl">
          <img src={Logo} width="100px" height="100px" />
          <h1 class="text--bold" style={{fontSize: "50px" ,marginBottom: "0"}}>Forest Guard</h1>
          <h1 class="text--normal" >Want to learn more?</h1>
          <div style={{
            display: "flex",
            justifyContent: "center",
          }}>
            <button onClick={() => location.href="../blog"} type="button" class="fg-button button button--lg margin-horiz--sm" style={{
              backgroundColor: "white",
              color: "#50806C",
              border: "solid",
              outlineColor: "#50806C",
              outlineWidth: "5px",
            }}>About</button>
            <button onClick={() => location.href="../docs/forest-guard-cli/installation"} type="button" class="fg-button button button--lg margin-horiz--sm" style={{
              backgroundColor: "#50806C",
            }}>Documentation</button>
          </div>
        </div>
      </section>
      <section class="fg-main-mobile">
        <div style={{
          textAlign: "center",
          backgroundClip: "white",
        }}>
          <MainTree class="margin-top--lg" height="auto" width="300px" />
          <h1 class="text--bold fg-title-text margin-bottom--md" style={{ fontSize: "50px" }}> Download </h1>
        </div>
        <div style={{
          backgroundColor: "#F9F9F9",
        }} class="padding-vert--lg padding-horiz--md">
          <div style={{
            backgroundColor: "white",
            textAlign: "center",
          }} class="fg-text-hero margin-horiz--md shadow--tl padding-horiz--lg padding-vert--lg">
            <h1 class="" style={{ color: "#232020", fontSize: "30px" }}>GUI Client</h1>
            <button onClick={() => location.href = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"} type="button" class="fg-button button button--lg margin-bottom--sm" style={{
              display: "block",
              margin: "auto",
              color: "white",
              fontSize: "25px",
              borderRadius: "20px",
            }}>
              Windows
            </button>
            <button onClick={() => location.href = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"} type="button" class="fg-button button button--lg margin-bottom--md" style={{
              display: "block",
              margin: "auto",
              color: "white",
              fontSize: "25px",
              borderRadius: "20px",
            }}>
              MacOS
            </button>
            <h5 class="" style={{ color: "#232020" }}>bro wanted exe file lmao 💀💀💀</h5>
            <div style={{
              display: "flex",
              justifyContent: "center",
            }} class="margin-top--lg">
              <Win class="margin-horiz--xs" height="50px" width="auto" />
              <Mac class="margin-horiz--xs" height="50px" width="auto" />
            </div>
          </div>
          <div style={{
            backgroundColor: "white",
            textAlign: "center",
          }} class="fg-text-hero margin-horiz--md margin-top--lg shadow--tl padding-horiz--lg padding-vert--lg">
            <h1 class="" style={{ color: "#232020", fontSize: "30px" }}>Python Package (CLI)</h1>
            <h3 class="margin-vert--sm" style={{ color: "#232020" }}>Run in your terminal</h3>
            <button onClick={() => location.href = "../docs/forest-guard-cli/installation"} type="button" class="fg-button button button--lg" style={{
              display: "block",
              margin: "auto",
              color: "white",
              fontSize: "17px",
            }}>
              $ pip install fguard
            </button>
            <h4 class="margin-vert--sm" style={{ color: "#232020" }}>Beta version</h4>
            <h3 class="margin-vert--sm" style={{ color: "#232020" }}>It is recommended to
              use <span style={{ color: "#6A994E" }}>local python virtual environment</span></h3>
            <div style={{
              display: "flex",
              justifyContent: "center",
            }} class="margin-top--lg">
              <Win class="margin-horiz--xs" height="50px" width="auto" />
              <Mac class="margin-horiz--xs" height="50px" width="auto" />
              <Ubuntu class="margin-horiz--xs" height="50px" width="auto" />
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
            <button onClick={() => location.href = "../docs/forest-guard-cli/installation"} type="button" class="fg-button button button--lg margin-bottom--sm" style={{
              backgroundColor: "#50806C",
              display: "block",
              margin: "auto",
            }}>Documentation</button>
            <button onClick={() => location.href = "../blog"} type="button" class="fg-button button button--lg" style={{
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
