#ifndef __TIMESTEPCONTROLLER_h__
#define __TIMESTEPCONTROLLER_h__

#include "Common/Common.h"
#include "TimeStep.h"
#include "SimulationModel.h"
#include "CollisionDetection.h"
#include <functional>

namespace PBD
{
	class TimeStepController : public TimeStep
	{
	public: 		
// 		static int SOLVER_ITERATIONS;
// 		static int SOLVER_ITERATIONS_V;
		static int NUM_SUB_STEPS;
		static int MAX_ITERATIONS;
		static int MAX_ITERATIONS_V;
		static int VELOCITY_UPDATE_METHOD;

		static int ENUM_VUPDATE_FIRST_ORDER;
		static int ENUM_VUPDATE_SECOND_ORDER;

	protected:
		int m_velocityUpdateMethod;
		unsigned int m_iterations;
		unsigned int m_iterationsV;
		unsigned int m_subSteps;
		unsigned int m_maxIterations;
		unsigned int m_maxIterationsV;
		std::function<void(ParticleData&)> m_externalAccelerationFunc;

		virtual void initParameters();
		
		void positionConstraintProjection(SimulationModel &model);
		void velocityConstraintProjection(SimulationModel &model);


	public:
		TimeStepController();
		virtual ~TimeStepController(void);

		virtual void step(SimulationModel &model);
		virtual void reset();

		// Experimental: allow runtime override of iteration counts (used by automation scripts)
		void setMaxIterations(unsigned int v) { m_maxIterations = v; }
		void setMaxIterationsV(unsigned int v) { m_maxIterationsV = v; }
		unsigned int getMaxIterations() const { return m_maxIterations; }
		unsigned int getMaxIterationsV() const { return m_maxIterationsV; }
		void setSubSteps(unsigned int v) { m_subSteps = v; }
		unsigned int getSubSteps() const { return m_subSteps; }
		void setExternalAccelerationFunc(const std::function<void(ParticleData&)> &f) { m_externalAccelerationFunc = f; }
	};
}

#endif
