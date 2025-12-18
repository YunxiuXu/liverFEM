#include "Experiment1XPBD.h"
#include "Simulation/Simulation.h"
#include "Simulation/TimeManager.h"
#include "Utils/Logger.h"

using namespace PBD;

namespace Exp1XPBD
{
	void (*resetFunc)() = nullptr;
	DemoBase *base = nullptr;

	static bool s_pulling = false;

	void init()
	{
		s_pulling = false;
	}

	void startPull()
	{
		if (resetFunc)
			resetFunc(); // like pressing 'r'
		if (base)
			base->setValue(DemoBase::PAUSE, false); // like pressing Space (run)
		s_pulling = true;
	}

	void stopPull()
	{
		s_pulling = false;
	}

	bool isPulling()
	{
		return s_pulling;
	}

	void step()
	{
		// no-op for now
	}

	std::function<void(ParticleData&)> externalAccelFunc()
	{
		return [](ParticleData &pd) {
			if (!s_pulling)
				return;

			// Match gravity application: add an acceleration of the same magnitude as |g| along +X.
			Simulation *sim = Simulation::getCurrent();
			const Vector3r grav(sim->getVecValue<Real>(Simulation::GRAVITATION));
			const Real a = std::abs(grav[1]) > static_cast<Real>(1e-9) ? std::abs(grav[1]) : grav.norm();
			if (a <= static_cast<Real>(0.0))
				return;

			for (unsigned int i = 0; i < pd.size(); ++i)
			{
				if (pd.getMass(i) == 0.0)
					continue;
				Vector3r &acc = pd.getAcceleration(i);
				acc[0] += a;
			}
		};
	}

	std::string status()
	{
		return s_pulling ? "Pulling (+X = |g|)" : "Idle";
	}
}

