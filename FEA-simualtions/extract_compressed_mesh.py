import vtk
import sys

def main(argv):
    if len(argv) < 3:
        print("Usage:", argv[0], "original_mesh compressed_mesh")
        return 1  # Return non-zero to indicate failure
    else:
        # Get filenames from command-line arguments
        original_mesh = sys.argv[1]
        output_mesh = sys.argv[2]

        # Read the original mesh
        meshReader = vtk.vtkUnstructuredGridReader()
        meshReader.SetFileName(original_mesh)
        meshReader.Update()

        # Extract initial points
        initial_points = []
        i_points = meshReader.GetOutput().GetPoints()
        NoP = meshReader.GetOutput().GetNumberOfPoints()

        # Temporary array to store point coordinates
        temp_point = [0.0, 0.0, 0.0]

        # Try-except block for error handling
        try:
            print("If the software fails here, please, check the node displacements")
            # Get displacements data array
            BoundCond = meshReader.GetOutput().GetPointData().GetScalars("displacements")
        except Exception as e:
            print("Error occurred while getting displacements:", str(e))
            exit()

        # Initialize final points list
        final_points = []

        # Iterate over each point
        for i in range(NoP):
            # Get coordinates of the point
            i_points.GetPoint(i, temp_point)
            
            # Store initial point coordinates
            initial_points.extend(temp_point)
            
            # Get displacement vector
            disp = BoundCond.GetTuple(i)
            
            # Calculate final point coordinates after displacement
            final_points.append(temp_point[0] + disp[0])
            final_points.append(temp_point[1] + disp[1])
            final_points.append(temp_point[2] + disp[2])

        # Elements
        accumm = 0
        minimum = 999999999
        elements = []

        i_cells = meshReader.GetOutput().GetCells()
        pts = vtk.vtkIdList()

        for i in range(i_cells.GetNumberOfCells()):
            i_cells.GetCell(accumm, pts)
            for j in range(pts.GetNumberOfIds()):
                elements.append(pts.GetId(j))
            accumm = accumm + 1 + pts.GetNumberOfIds()

        # Create Unstructured Grid
        u_grid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()

        points.SetNumberOfPoints(NoP)
        for i in range(NoP):
            temp_point = final_points[3*i:3*(i+1)]
            points.SetPoint(i, temp_point)

        u_grid.SetPoints(points)
        u_grid.SetCells(vtk.VTK_TETRA, i_cells)

        # Write moved mesh
        meshWriter = vtk.vtkUnstructuredGridWriter()
        meshWriter.SetInputData(u_grid)
        meshWriter.SetFileName(output_mesh)
        meshWriter.Update()

        print("Compressed mesh is obtained!")

        return 0  # Return zero to indicate success

if __name__ == "__main__":
    sys.exit(main(sys.argv))