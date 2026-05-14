import { useEffect, useState } from "react";

import ForceGraph2D from "react-force-graph-2d";


function KnowledgeGraph() {

  const [graphData, setGraphData] = useState({
    nodes: [],
    links: []
  });


  // =====================================
  // Fetch Graph
  // =====================================

  const fetchGraph = async () => {

    try {

      const response = await fetch(
        "http://127.0.0.1:8000/knowledge-graph"
      );

      const data = await response.json();

      console.log(
        "Knowledge Graph API:",
        data
      );

      if (!Array.isArray(data.graph)) {

        console.error(
          "Invalid graph data"
        );

        return;
      }

      const nodes = [];

      const links = [];

      const nodeSet = new Set();


      data.graph.forEach((item) => {

        // =========================
        // Entities
        // =========================

        if (item.entities) {

          item.entities.forEach((entity) => {

            if (!nodeSet.has(entity.text)) {

              nodes.push({

                id: entity.text,

                group: entity.label
              });

              nodeSet.add(entity.text);
            }

          });
        }


        // =========================
        // Relationships
        // =========================

        if (item.relationships) {

          item.relationships.forEach((relation) => {

            links.push({

              source: relation.subject,

              target: relation.object,

              label: relation.relation
            });


            if (!nodeSet.has(
                relation.subject
            )) {

              nodes.push({

                id: relation.subject,

                group: "RELATION"
              });

              nodeSet.add(
                relation.subject
              );
            }


            if (!nodeSet.has(
                relation.object
            )) {

              nodes.push({

                id: relation.object,

                group: "RELATION"
              });

              nodeSet.add(
                relation.object
              );
            }

          });
        }

      });


      console.log(
        "NODES:",
        nodes
      );

      console.log(
        "LINKS:",
        links
      );

      setGraphData({
        nodes,
        links
      });

    } catch (error) {

      console.error(
        "Graph Error:",
        error
      );
    }
  };


  // =====================================
  // Initial Load
  // =====================================

  useEffect(() => {

    fetchGraph();

  }, []);


  return (

    <div>

      {/* Header */}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px"
        }}
      >

        <h2>
          NeuralWiki Knowledge Graph
        </h2>


        <button

          onClick={fetchGraph}

          style={{
            padding: "10px 20px",
            background: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer"
          }}
        >
          Refresh Graph
        </button>

      </div>


      {/* Graph */}

      <div
        style={{
          border: "1px solid #ccc",
          borderRadius: "10px",
          overflow: "hidden",
          background: "white"
        }}
      >

        <ForceGraph2D

          graphData={graphData}

          nodeLabel="id"

          linkLabel="label"

          width={1000}

          height={700}

          nodeAutoColorBy="group"

          backgroundColor="#ffffff"
        />

      </div>

    </div>
  );
}

export default KnowledgeGraph;