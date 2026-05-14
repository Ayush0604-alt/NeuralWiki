import Upload from "./pages/Upload";
import SearchBox from "./components/SearchBox";
import ChatBox from "./components/ChatBox";
import KnowledgeGraph from "./components/KnowledgeGraph";

function App() {

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "40px",
        padding: "40px",
        fontFamily: "Arial"
      }}
    >

      <div
        style={{
          border: "1px solid #ccc",
          padding: "20px",
          borderRadius: "10px"
        }}
      >
        <Upload />
      </div>

      <div
        style={{
          border: "1px solid #ccc",
          padding: "20px",
          borderRadius: "10px"
        }}
      >
        <SearchBox />
      </div>

      <div
        style={{
          border: "1px solid #ccc",
          padding: "20px",
          borderRadius: "10px"
        }}
      >
        <ChatBox />
        <div
  style={{
    border: "1px solid #ccc",
    padding: "20px",
    borderRadius: "10px"
  }}
>
  <KnowledgeGraph />
</div>
      </div>

    </div>
  );
}

export default App;